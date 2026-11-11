// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use vello_encoding::{
    BinHeader, BumpAllocators, ConfigUniform, DrawMonoid, GLASS_UNCLIPPED_BIN, PathBbox,
};

use super::CpuBinding;

const WG_SIZE: usize = 256;
const TILE_WIDTH: usize = 16;
const TILE_HEIGHT: usize = 16;
const N_TILE_X: usize = 16;
const N_TILE_Y: usize = 16;
const SX: f32 = 1.0 / ((N_TILE_X * TILE_WIDTH) as f32);
const SY: f32 = 1.0 / ((N_TILE_Y * TILE_HEIGHT) as f32);

fn bbox_intersect(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

fn binning_main(
    n_wg: u32,
    config: &ConfigUniform,
    draw_monoids: &[DrawMonoid],
    path_bbox_buf: &[PathBbox],
    clip_bbox_buf: &[[f32; 4]],
    intersected_bbox: &mut [[f32; 4]],
    bump: &mut BumpAllocators,
    bin_data: &mut [u32],
    bin_header: &mut [BinHeader],
) {
    let width_in_bins = config.width_in_tiles.div_ceil(N_TILE_X as u32) as i32;
    let height_in_bins = config.height_in_tiles.div_ceil(N_TILE_Y as u32) as i32;
    let n_bins = (width_in_bins * height_in_bins) as usize;
    for wg in 0..n_wg as usize {
        let mut counts = vec![0; n_bins];
        let mut bboxes = [[0, 0, 0, 0]; WG_SIZE];
        for local_ix in 0..WG_SIZE {
            let element_ix = wg * WG_SIZE + local_ix;
            let mut x0 = 0;
            let mut y0 = 0;
            let mut x1 = 0;
            let mut y1 = 0;
            if element_ix < config.layout.n_draw_objects as usize {
                let draw_monoid = draw_monoids[element_ix];
                let mut clip_bbox = [-1e9, -1e9, 1e9, 1e9];
                if draw_monoid.clip_ix > 0 {
                    assert!(draw_monoid.clip_ix - 1 < config.layout.n_clips);
                    clip_bbox = clip_bbox_buf[draw_monoid.clip_ix as usize - 1];
                }
                let path_bbox = path_bbox_buf[draw_monoid.path_ix as usize];
                let pb = [
                    path_bbox.x0 as f32,
                    path_bbox.y0 as f32,
                    path_bbox.x1 as f32,
                    path_bbox.y1 as f32,
                ];
                let mut bbox = bbox_intersect(clip_bbox, pb);
                // A glass draw is binned over its own box, and coarse gets the clip box in
                // tiles. An END_CLIP's info offset is its BEGIN_CLIP's (clip_leaf), and a draw
                // past the info stream has no info.
                let di = draw_monoid.info_offset as usize;
                if di < config.layout.bin_data_start as usize
                    && bin_data[di] & GLASS_UNCLIPPED_BIN != 0
                {
                    let (w, h) = (config.width_in_tiles as f32, config.height_in_tiles as f32);
                    let clip_x0 = (clip_bbox[0] / TILE_WIDTH as f32).floor().clamp(0.0, w) as u32;
                    let clip_y0 = (clip_bbox[1] / TILE_HEIGHT as f32).floor().clamp(0.0, h) as u32;
                    let clip_x1 = (clip_bbox[2] / TILE_WIDTH as f32).ceil().clamp(0.0, w) as u32;
                    let clip_y1 = (clip_bbox[3] / TILE_HEIGHT as f32).ceil().clamp(0.0, h) as u32;
                    bin_data[di + 1] = clip_x0 | (clip_y0 << 16);
                    bin_data[di + 2] = clip_x1 | (clip_y1 << 16);
                    bbox = pb;
                }
                intersected_bbox[element_ix] = bbox;
                if bbox[0] < bbox[2] && bbox[1] < bbox[3] {
                    x0 = (bbox[0] * SX).floor() as i32;
                    y0 = (bbox[1] * SY).floor() as i32;
                    x1 = (bbox[2] * SX).ceil() as i32;
                    y1 = (bbox[3] * SY).ceil() as i32;
                }
            }
            x0 = x0.clamp(0, width_in_bins);
            y0 = y0.clamp(0, height_in_bins);
            x1 = x1.clamp(0, width_in_bins);
            y1 = y1.clamp(0, height_in_bins);
            for y in y0..y1 {
                for x in x0..x1 {
                    counts[(y * width_in_bins + x) as usize] += 1;
                }
            }
            bboxes[local_ix] = [x0, y0, x1, y1];
        }
        let mut chunk_offset = vec![0; n_bins];
        let aligned_n_bins = (n_bins + 255) & !255;
        // Looping on the number of bins instead of workgroup sized chunks, not idiomatic of gpus
        // Oh well!
        for local_ix in 0..n_bins {
            let global_ix = wg * aligned_n_bins + local_ix;
            chunk_offset[local_ix] = bump.binning;
            bump.binning += counts[local_ix];
            bin_header[global_ix] = BinHeader {
                element_count: counts[local_ix],
                chunk_offset: chunk_offset[local_ix],
            };
        }
        for local_ix in 0..WG_SIZE {
            let element_ix = wg * WG_SIZE + local_ix;
            let bbox = bboxes[local_ix];
            for y in bbox[1]..bbox[3] {
                for x in bbox[0]..bbox[2] {
                    let bin_ix = (y * width_in_bins + x) as usize;
                    let ix = config.layout.bin_data_start + chunk_offset[bin_ix];
                    bin_data[ix as usize] = element_ix as u32;
                    chunk_offset[bin_ix] += 1;
                }
            }
        }
    }
}

pub fn binning(n_wg: u32, resources: &[CpuBinding<'_>]) {
    let config = resources[0].as_typed();
    let draw_monoids = resources[1].as_slice();
    let path_bbox_buf = resources[2].as_slice();
    let clip_bbox_buf = resources[3].as_slice();
    let mut intersected_bbox = resources[4].as_slice_mut();
    let mut bump = resources[5].as_typed_mut();
    let mut bin_data = resources[6].as_slice_mut();
    let mut bin_header = resources[7].as_slice_mut();
    binning_main(
        n_wg,
        &config,
        &draw_monoids,
        &path_bbox_buf,
        &clip_bbox_buf,
        &mut intersected_bbox,
        &mut bump,
        &mut bin_data,
        &mut bin_header,
    );
}
