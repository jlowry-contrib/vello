// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glass markers: points in paint order where glass reads its backdrop.

pub use vello_encoding::{
    GlassMarker, GlassMarkers, GlassSkip, IRect, MAX_GLASS_LAYERS, MAX_GLASS_PHASES, assign_phases,
    pack_shelves,
};

#[cfg(feature = "wgpu")]
use crate::render::{glass_record_words, glass_state_size};
#[cfg(feature = "wgpu")]
use crate::wgpu_engine::pooled_size;
#[cfg(feature = "wgpu")]
use vello_encoding::assign_phases_fitting;

#[cfg(feature = "wgpu")]
use wgpu::{CommandEncoder, Device, Queue, Texture};

/// Where a glass marker's backdrop and glass go in this frame's atlases.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MarkerPlacement {
    /// The marker's identifier.
    pub id: u32,
    /// The phase whose passes produce this marker's glass.
    pub phase: u32,
    /// Pixels whose backdrop the glass reads, tile-snapped and clamped to the target.
    pub region: IRect,
    /// Pixels the glass writes, tile-snapped and clamped to the target.
    pub composite: IRect,
    /// Top-left of the marker's slot in the backdrop atlas.
    pub backdrop_origin: (u32, u32),
    /// Top-left of the marker's slot in the glass atlas.
    pub glass_origin: (u32, u32),
}

/// Why a frame leaves out a marker that has pixels on the target.
///
/// Only the renderer's logs report it (`glass.phase.capped`, `glass.state.full`).
#[cfg(feature = "wgpu")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum GlassLeftOut {
    /// The marker would need a phase past [`MAX_GLASS_PHASES`].
    PhaseCap,
    /// The marker doesn't fit the device: its resume records would take the glass state past
    /// the storage binding limit, a list past the dispatch limit, or an atlas past the
    /// texture size limit.
    StateFull,
}

/// How fine saves its colour state when a tile stops for glass.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum GlassResumeFormat {
    /// Four `f32` per colour: exact.
    #[default]
    F32,
    /// Four `unorm16` per colour: half the bandwidth.
    Unorm16,
}

/// Texel format of the glass atlas the external passes write.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum GlassAtlasFormat {
    /// Half-float premultiplied colour.
    #[default]
    Rgba16Float,
    /// 8-bit premultiplied colour.
    Rgba8Unorm,
}

/// Everything the external passes learn about a frame before any GPU work is recorded.
#[derive(Copy, Clone, Debug)]
pub struct GlassFrame<'a> {
    /// Every placed marker, ordered by phase and then paint order.
    pub markers: &'a [MarkerPlacement],
    /// Number of phases; the passes are called for phases `1..=phases`.
    pub phases: u32,
}

/// One phase's resources, handed to [`GlassPasses::record`].
///
/// The atlases persist across frames, so a backdrop slot's pixels outside the marker's
/// region intersected with its clip are undefined, as are glass slot pixels the passes
/// don't write.
#[cfg(feature = "wgpu")]
#[derive(Copy, Clone, Debug)]
pub struct GlassPhase<'a> {
    /// The phase index, from 1.
    pub index: u32,
    /// Premultiplied `Rgba8Unorm` backdrops, one slot per marker.
    pub backdrop_atlas: &'a Texture,
    /// Premultiplied glass, one slot per marker, for fine to composite.
    pub glass_atlas: &'a Texture,
    /// The markers of this phase, in paint order.
    pub markers: &'a [MarkerPlacement],
}

/// Passes that turn backdrops into glass, recorded into Vello's command encoder.
///
/// The atlas textures can be recreated on any frame, as they grow or shrink, so hooks must
/// not keep atlas textures, views or bind groups from one frame for the next.
///
/// The hook is a trait object (M-DI-HIERARCHY) because Vello can't name the types of
/// the crate that implements the glass passes.
#[cfg(feature = "wgpu")]
pub trait GlassPasses {
    /// Sees every marker of the frame before any GPU work is recorded.
    fn prepare(&mut self, device: &Device, queue: &Queue, frame: &GlassFrame<'_>);
    /// Records the passes that produce phase `phase.index`'s glass atlas.
    fn record(&mut self, device: &Device, encoder: &mut CommandEncoder, phase: &GlassPhase<'_>);
}

/// One row of fine's marker table; mirrors `GlassMarker` in `fine.wgsl`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct GlassMarkerRow {
    /// Region in tiles: x0, y0, x1, y1.
    region: [u32; 4],
    /// Composite in tiles: x0, y0, x1, y1.
    composite: [u32; 4],
    /// 0 for a marker the frame leaves out.
    phase: u32,
    /// Backdrop slot origin, `x | y << 16`.
    backdrop_origin: u32,
    /// Glass slot origin, `x | y << 16`.
    glass_origin: u32,
    pad: u32,
}

/// A frame's glass layout: placements, the marker table and atlas sizes.
#[cfg(feature = "wgpu")]
#[derive(Debug, Default)]
pub(crate) struct GlassPlan {
    /// Placed markers, ordered by phase and then paint order.
    pub(crate) placements: Vec<MarkerPlacement>,
    /// One row per encoded marker, indexed by the marker index in the draw data.
    pub(crate) rows: Vec<GlassMarkerRow>,
    pub(crate) phases: u32,
    pub(crate) backdrop_size: (u32, u32),
    pub(crate) glass_size: (u32, u32),
    /// Most tiles with glass commands: every kept marker's region tiles on the target, at
    /// most the target's tiles.
    pub(crate) list_capacity: u32,
    /// Most stops each phase can see: its markers' composite tiles.
    pub(crate) capacities: [u32; MAX_GLASS_PHASES as usize],
    /// Size of each phase's resume records, in words, from the deepest of its markers.
    pub(crate) record_words: [u32; MAX_GLASS_PHASES as usize],
    /// State-full markers whose records would pass the size limit.
    pub(crate) over_size: usize,
    /// State-full markers whose tiles would pass the dispatch limit; a marker can pass more
    /// than one limit.
    pub(crate) over_dispatch: usize,
    /// State-full markers whose slots would take an atlas past the texture size limit.
    pub(crate) over_atlas: usize,
    /// Why the frame leaves out each encoded marker, if it does and the marker has pixels on
    /// the target under at most [`MAX_GLASS_LAYERS`] open layers.
    pub(crate) left_out: Vec<Option<GlassLeftOut>>,
    /// Draw data offsets of every encoded marker the frame leaves out.
    pub(crate) left_out_offsets: Vec<usize>,
}

#[cfg(feature = "wgpu")]
impl GlassPlan {
    /// Returns how many markers the glass state's limits left out.
    pub(crate) fn state_full(&self) -> usize {
        self.left_out
            .iter()
            .filter(|&&l| l == Some(GlassLeftOut::StateFull))
            .count()
    }
}

/// Device limits a frame's glass must fit in.
#[cfg(feature = "wgpu")]
#[derive(Copy, Clone, Debug)]
pub(crate) struct GlassLimits {
    /// Largest `glass_state` buffer the device binds, in bytes.
    pub(crate) state_size: u64,
    /// Most workgroups in one dimension of a dispatch: list 0's and each phase's tiles are
    /// dispatched in one dimension.
    pub(crate) workgroups: u32,
    /// Resume colour format, 0 for `F32` and 1 for `Unorm16`.
    pub(crate) resume_color: u32,
    /// Longest side of an atlas texture, in pixels. Neither atlas format adds a limit of its
    /// own to the device's `max_texture_dimension_2d`.
    pub(crate) texture_side: u32,
}

/// Returns the placements of phase `index`, given placements ordered by phase.
#[cfg(feature = "wgpu")]
pub(crate) fn phase_placements(placements: &[MarkerPlacement], index: u32) -> &[MarkerPlacement] {
    let start = placements.partition_point(|p| p.phase < index);
    let end = placements.partition_point(|p| p.phase <= index);
    &placements[start..end]
}

/// Lays out a frame that runs the glass permutation with no glass: a zeroed marker table
/// with a row per encoded marker (at least one, so the table binds), no phase, and 1×1
/// atlases.
///
/// The atlases keep their size until a run of small frames shrinks them, and the pooled
/// `glass_state` is reused, so after an earlier glass frame the forced frame binds those
/// larger resources. Without markers, fine samples neither atlas and reads only the state
/// header.
#[cfg(feature = "wgpu")]
pub(crate) fn forced_plan(markers: usize) -> GlassPlan {
    GlassPlan {
        rows: vec![GlassMarkerRow::default(); markers.max(1)],
        backdrop_size: (1, 1),
        glass_size: (1, 1),
        ..GlassPlan::default()
    }
}

/// Side of a fine tile in pixels, matching `TILE_WIDTH` and `TILE_HEIGHT` in `config.wgsl`.
#[cfg(feature = "wgpu")]
const TILE: i32 = 16;

/// Narrowest atlas the shelf packer makes; one 512 px shelf holds most frames' slots.
#[cfg(feature = "wgpu")]
const MIN_ATLAS_WIDTH: u32 = 512;

/// Lays out a frame's glass for a `width`×`height` target within `limits`.
///
/// Markers under more than [`MAX_GLASS_LAYERS`] open layers, with no pixels on the
/// target, past the phase cap, past the glass state's limits or whose slots would take an
/// atlas past the texture size limit are left out: their rows have phase 0 and they get no
/// placement. Both atlas sizes the plan returns fit in `limits.texture_side`. Markers are kept in paint order while they fit;
/// one that doesn't is left out and later ones are still tried.
#[cfg(feature = "wgpu")]
pub(crate) fn plan(
    markers: &GlassMarkers,
    width: u32,
    height: u32,
    limits: GlassLimits,
) -> GlassPlan {
    let tiles = |r: IRect| {
        let r = r.snap_out(TILE);
        [r.x0, r.y0, r.x1, r.y1].map(|v| (v / TILE) as u32)
    };
    let size = |r: IRect| {
        let r = r.snap_out(TILE);
        ((r.x1 - r.x0) as u32, (r.y1 - r.y0) as u32)
    };
    let tile_count = |r: IRect| {
        let [x0, y0, x1, y1] = tiles(r);
        (x1 - x0) * (y1 - y0)
    };
    let target_tiles = width.div_ceil(TILE as u32) * height.div_ceil(TILE as u32);
    let mut included = Vec::new();
    let mut clamped = Vec::new();
    for (i, (marker, &depth)) in markers.markers.iter().zip(&markers.depths).enumerate() {
        let region = marker.region.clamp(width, height);
        let composite = marker.composite.clamp(width, height);
        if depth <= MAX_GLASS_LAYERS && !region.is_empty() && !composite.is_empty() {
            included.push(i);
            clamped.push(GlassMarker {
                id: marker.id,
                region,
                composite,
            });
        }
    }
    let mut left_out = vec![None; markers.len()];
    if included.is_empty() {
        return GlassPlan {
            left_out,
            ..GlassPlan::default()
        };
    }
    // Coarse writes glass commands over the regions of the kept markers only, so list 0
    // holds at most their tiles, and at most the target's.
    let mut region_tiles: u64 = 0;
    let mut list_capacity: u32 = 0;
    let mut capacities = [0_u32; MAX_GLASS_PHASES as usize];
    let mut depths = [0; MAX_GLASS_PHASES as usize];
    // Whether the state's limits rejected each included marker.
    let mut state_full = vec![false; clamped.len()];
    let (mut over_size, mut over_dispatch, mut over_atlas) = (0, 0, 0);
    // Slot sizes of the markers kept so far, in paint order, as the final packing sees them.
    let (mut backdrop_sizes, mut glass_sizes) = (Vec::new(), Vec::new());
    let fits_atlas = |sizes: &[(u32, u32)]| {
        let (_, (w, h)) = pack_shelves(sizes, MIN_ATLAS_WIDTH, limits.texture_side);
        w <= limits.texture_side && h <= limits.texture_side
    };
    let (phases, capped) = assign_phases_fitting(&clamped, |j, phase| {
        let p = phase as usize - 1;
        let regions = region_tiles + u64::from(tile_count(clamped[j].region));
        let list = regions.min(u64::from(target_tiles)) as u32;
        let mut caps = capacities;
        caps[p] = caps[p].saturating_add(tile_count(clamped[j].composite));
        let mut deepest = depths;
        deepest[p] = deepest[p].max(markers.depths[included[j]]);
        let words = deepest.map(|d| glass_record_words(limits.resume_color, d));
        let state_size = pooled_size(glass_state_size(list, caps, words));
        let fits_size = state_size <= limits.state_size;
        let fits_dispatch = list <= limits.workgroups && caps[p] <= limits.workgroups;
        backdrop_sizes.push(size(clamped[j].region));
        glass_sizes.push(size(clamped[j].composite));
        let fits_atlases = fits_atlas(&backdrop_sizes) && fits_atlas(&glass_sizes);
        let fits = fits_size && fits_dispatch && fits_atlases;
        if fits {
            (region_tiles, list_capacity, capacities, depths) = (regions, list, caps, deepest);
        } else {
            backdrop_sizes.pop();
            glass_sizes.pop();
            state_full[j] = true;
            over_size += usize::from(!fits_size);
            over_dispatch += usize::from(!fits_dispatch);
            over_atlas += usize::from(!fits_atlases);
        }
        fits
    });
    if capped > 0 {
        log::warn!("glass.phase.capped markers={capped} max={MAX_GLASS_PHASES}");
    }
    for (j, &phase) in phases.iter().enumerate() {
        if phase == 0 {
            left_out[included[j]] = Some(if state_full[j] {
                GlassLeftOut::StateFull
            } else {
                GlassLeftOut::PhaseCap
            });
        }
    }
    // (encoded index, clamped marker, phase) of each marker the frame places.
    let kept: Vec<_> = included
        .into_iter()
        .zip(clamped)
        .zip(phases)
        .filter(|&(_, phase)| phase != 0)
        .map(|((i, marker), phase)| (i, marker, phase))
        .collect();
    let (backdrop_origins, backdrop_size) =
        pack_shelves(&backdrop_sizes, MIN_ATLAS_WIDTH, limits.texture_side);
    let (glass_origins, glass_size) =
        pack_shelves(&glass_sizes, MIN_ATLAS_WIDTH, limits.texture_side);
    let pack = |(x, y): (u32, u32)| x | y << 16;
    let mut rows = vec![GlassMarkerRow::default(); markers.len()];
    let mut placements = Vec::with_capacity(kept.len());
    for (j, &(i, marker, phase)) in kept.iter().enumerate() {
        rows[i] = GlassMarkerRow {
            region: tiles(marker.region),
            composite: tiles(marker.composite),
            phase,
            backdrop_origin: pack(backdrop_origins[j]),
            glass_origin: pack(glass_origins[j]),
            pad: 0,
        };
        placements.push(MarkerPlacement {
            id: marker.id,
            phase,
            region: marker.region,
            composite: marker.composite,
            backdrop_origin: backdrop_origins[j],
            glass_origin: glass_origins[j],
        });
    }
    placements.sort_by_key(|p| p.phase);
    let left_out_offsets = rows
        .iter()
        .zip(&markers.draw_data_offsets)
        .filter(|(row, _)| row.phase == 0)
        .map(|(_, &offset)| offset)
        .collect();
    GlassPlan {
        phases: placements.last().map_or(0, |p| p.phase),
        placements,
        rows,
        backdrop_size,
        glass_size,
        list_capacity,
        capacities,
        record_words: depths.map(|d| glass_record_words(limits.resume_color, d)),
        over_size,
        over_dispatch,
        over_atlas,
        left_out,
        left_out_offsets,
    }
}

/// Frames an atlas must need at most a quarter of its area before it shrinks: 2 s at 60 Hz.
#[cfg(feature = "wgpu")]
const ATLAS_SHRINK_FRAMES: u32 = 120;

/// Sizes a glass atlas across frames: it grows at once to what a frame needs, and shrinks
/// after [`ATLAS_SHRINK_FRAMES`] consecutive frames needing at most a quarter of its area.
#[cfg(feature = "wgpu")]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct AtlasSizer {
    size: (u32, u32),
    /// Consecutive frames that needed at most a quarter of the area.
    low_frames: u32,
    /// The largest size those frames needed.
    low_need: (u32, u32),
}

#[cfg(feature = "wgpu")]
impl AtlasSizer {
    /// Returns the atlas size for a frame that needs `need`, if neither side passes `max`.
    ///
    /// # Errors
    ///
    /// Returns the size that would pass `max`, leaving the sizer as it was, so a later
    /// frame that fits isn't held to it.
    pub(crate) fn fit(&mut self, need: (u32, u32), max: u32) -> Result<(u32, u32), (u32, u32)> {
        let mut next = *self;
        let size = next.size(need);
        if size.0 > max || size.1 > max {
            return Err(size);
        }
        *self = next;
        Ok(size)
    }

    /// Returns the atlas size for a frame that needs `need`, with no limit.
    ///
    /// A shrink goes to the smallest power-of-two size that fits every frame of the low run.
    fn size(&mut self, need: (u32, u32)) -> (u32, u32) {
        let area = |(w, h): (u32, u32)| u64::from(w) * u64::from(h);
        let grown = (self.size.0.max(need.0), self.size.1.max(need.1));
        if grown != self.size {
            *self = Self {
                size: grown,
                ..Self::default()
            };
        } else if 4 * area(need) <= area(self.size) {
            self.low_frames += 1;
            self.low_need = (self.low_need.0.max(need.0), self.low_need.1.max(need.1));
            if self.low_frames >= ATLAS_SHRINK_FRAMES {
                let fit = |v: u32| v.max(1).next_power_of_two();
                *self = Self {
                    size: (fit(self.low_need.0), fit(self.low_need.1)),
                    ..Self::default()
                };
            }
        } else {
            self.low_frames = 0;
            self.low_need = (0, 0);
        }
        self.size
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::{
        AtlasSizer, GlassLeftOut, GlassLimits, GlassMarker, GlassMarkers, GlassPlan, IRect,
        phase_placements,
    };
    use crate::render::glass_state_size;

    /// The default WebGPU limits: 128 MiB bindings and 65535 workgroups per dimension.
    const LIMITS: GlassLimits = GlassLimits {
        state_size: 128 << 20,
        workgroups: 65535,
        resume_color: 0,
        texture_side: 8192,
    };

    fn plan(markers: &GlassMarkers, width: u32, height: u32) -> GlassPlan {
        super::plan(markers, width, height, LIMITS)
    }

    fn state_size(plan: &GlassPlan) -> u64 {
        glass_state_size(plan.list_capacity, plan.capacities, plan.record_words)
    }

    fn r(x0: i32, y0: i32, x1: i32, y1: i32) -> IRect {
        IRect { x0, y0, x1, y1 }
    }

    /// Markers as the encoding stores them: already snapped to tiles.
    fn markers(list: &[(u32, IRect, IRect, u32)]) -> GlassMarkers {
        GlassMarkers {
            markers: list
                .iter()
                .map(|&(id, region, composite, _)| GlassMarker {
                    id,
                    region,
                    composite,
                })
                .collect(),
            draw_data_offsets: (0..list.len()).map(|i| 3 * i).collect(),
            depths: list.iter().map(|m| m.3).collect(),
        }
    }

    #[test]
    fn regions_clamp_to_the_target_and_slots_stay_whole_tiles() {
        let m = markers(&[(7, r(400, 0, 528, 48), r(416, 16, 480, 32), 0)]);
        let plan = plan(&m, 500, 500);
        assert_eq!(plan.phases, 1);
        assert_eq!(plan.placements.len(), 1);
        assert_eq!(plan.placements[0].id, 7);
        assert_eq!(plan.placements[0].region, r(400, 0, 500, 48));
        assert_eq!(plan.rows[0].region, [25, 0, 32, 3]);
        assert_eq!(plan.rows[0].composite, [26, 1, 30, 2]);
        assert_eq!(plan.capacities, [4, 0, 0, 0]);
        assert_eq!(plan.backdrop_size, (512, 64));
    }

    #[test]
    fn offscreen_and_too_deep_markers_get_phase_zero_rows() {
        let m = markers(&[
            (1, r(608, 0, 640, 32), r(608, 0, 624, 16), 0),
            (2, r(0, 0, 32, 32), r(0, 0, 16, 16), 9),
            (3, r(0, 0, 32, 32), r(0, 0, 16, 16), 8),
        ]);
        let plan = plan(&m, 512, 512);
        let ids: Vec<_> = plan.placements.iter().map(|p| p.id).collect();
        assert_eq!(ids, [3]);
        assert_eq!(plan.rows.len(), 3);
        assert_eq!(plan.rows[0].phase, 0);
        assert_eq!(plan.rows[1].phase, 0);
        assert_eq!(plan.rows[2].phase, 1);
    }

    #[test]
    fn markers_past_the_phase_cap_get_phase_zero_rows() {
        let chain: Vec<_> = (0..5)
            .map(|i| {
                let o = 64 * i;
                let region = r(64 + o, 64 + o, 192 + o, 192 + o);
                (i as u32, region, r(96 + o, 96 + o, 160 + o, 160 + o), 0)
            })
            .collect();
        let plan = plan(&markers(&chain), 512, 512);
        let ids: Vec<_> = plan.placements.iter().map(|p| (p.id, p.phase)).collect();
        assert_eq!(ids, [(0, 1), (1, 2), (2, 3), (3, 4)]);
        assert_eq!(plan.phases, 4);
        assert_eq!(plan.rows[4], super::GlassMarkerRow::default());
        assert_eq!(plan.capacities, [16; 4]);
    }

    #[test]
    fn markers_past_the_state_limit_are_left_out() {
        // Disjoint markers, region equal to composite, in 160 px cells: each composite tile
        // costs a depth-0 `F32` record of 1028 words, its phase 1 list word and its list 0
        // word, 4120 bytes. 1 MiB, a pool size class, holds the 32-byte header and 254 tiles.
        let sides = [8, 8, 8, 2, 8, 2, 4, 8, 1, 6, 1, 4, 3, 2, 1, 2];
        let list: Vec<_> = sides
            .iter()
            .enumerate()
            .map(|(i, &side)| {
                let (x, y) = (160 * (i as i32 % 4), 160 * (i as i32 / 4));
                let r = r(x, y, x + 16 * side, y + 16 * side);
                (i as u32, r, r, 0)
            })
            .collect();
        let limits = GlassLimits {
            state_size: 1 << 20,
            ..LIMITS
        };
        let plan = super::plan(&markers(&list), 640, 640, limits);
        // Running tile totals 64, 128, 192, 196, then 260 leaves marker 4 out, 200, 216, 280
        // leaves 7 out, 217, 253, 254, and every later marker would pass 254.
        let kept: Vec<_> = plan.placements.iter().map(|p| p.id).collect();
        assert_eq!(kept, [0, 1, 2, 3, 5, 6, 8, 9, 10]);
        for i in [4, 7, 11, 12, 13, 14, 15] {
            assert_eq!(
                plan.left_out[i],
                Some(GlassLeftOut::StateFull),
                "marker {i}"
            );
            assert_eq!(plan.rows[i].phase, 0, "marker {i}");
        }
        assert_eq!(plan.state_full(), 7);
        assert_eq!(
            (plan.over_size, plan.over_dispatch, plan.over_atlas),
            (7, 0, 0)
        );
        assert_eq!(plan.capacities, [254, 0, 0, 0]);
        assert_eq!(plan.list_capacity, 254);
        assert_eq!(state_size(&plan), 32 + 254 * 4120);
        assert_eq!(plan.left_out_offsets, [12, 21, 33, 36, 39, 42, 45]);
    }

    #[test]
    fn markers_past_the_dispatch_limit_are_left_out() {
        let list = [
            (0, r(0, 0, 64, 64), r(0, 0, 64, 64), 0),
            (1, r(128, 0, 160, 32), r(128, 0, 160, 32), 0),
            (2, r(256, 0, 272, 16), r(256, 0, 272, 16), 0),
        ];
        // 16 tiles, then 4 more would pass 19, then 1.
        let limits = GlassLimits {
            workgroups: 19,
            ..LIMITS
        };
        let plan = super::plan(&markers(&list), 512, 512, limits);
        let kept: Vec<_> = plan.placements.iter().map(|p| p.id).collect();
        assert_eq!(kept, [0, 2]);
        assert_eq!(plan.left_out[1], Some(GlassLeftOut::StateFull));
        assert_eq!(
            (plan.over_size, plan.over_dispatch, plan.over_atlas),
            (0, 1, 0)
        );
        assert_eq!(plan.list_capacity, 17);
    }

    #[test]
    fn markers_past_the_atlas_limit_are_left_out() {
        // Under a 1024 px texture limit, an atlas holds sixteen 256 px slots on four shelves.
        // After twelve of them, a 512 px marker would need a 512 px shelf of its own plus
        // three more, so it's left out, and the next four 256 px markers still fit.
        let mut list: Vec<_> = (0..17)
            .map(|i| {
                let r = r(256 * i, 0, 256 * (i + 1), 256);
                (i as u32, r, r, 0)
            })
            .collect();
        list.insert(12, (99, r(0, 512, 512, 1024), r(0, 512, 512, 1024), 0));
        let limits = GlassLimits {
            texture_side: 1024,
            ..LIMITS
        };
        let plan = super::plan(&markers(&list), 4352, 1024, limits);
        let kept: Vec<_> = plan.placements.iter().map(|p| p.id).collect();
        assert_eq!(kept, (0..16).collect::<Vec<_>>());
        for i in [12, 17] {
            assert_eq!(plan.left_out[i], Some(GlassLeftOut::StateFull), "{i}");
            assert_eq!(plan.rows[i].phase, 0, "{i}");
        }
        assert_eq!(
            (plan.over_size, plan.over_dispatch, plan.over_atlas),
            (0, 0, 2)
        );
        assert_eq!(plan.backdrop_size, (1024, 1024));
        assert_eq!(plan.glass_size, (1024, 1024));
        assert_eq!(plan.left_out_offsets, [36, 51]);
    }

    #[test]
    fn depth_zero_records_are_smaller() {
        let shallow = markers(&[(0, r(0, 0, 32, 32), r(0, 0, 16, 16), 0)]);
        let deep = markers(&[(0, r(0, 0, 32, 32), r(0, 0, 16, 16), 2)]);
        let (shallow, deep) = (plan(&shallow, 512, 512), plan(&deep, 512, 512));
        // Header, four list 0 words, one phase list word, then the record: 4 header words,
        // 1024 colour words and 256 words per open layer.
        assert_eq!(shallow.record_words[0], 1028);
        assert_eq!(deep.record_words[0], 1540);
        assert_eq!(state_size(&shallow), 32 + 4 * 4 + 4 * (1 + 1028));
        assert_eq!(state_size(&deep), 32 + 4 * 4 + 4 * (1 + 1540));
        let unorm16 = GlassLimits {
            resume_color: 1,
            ..LIMITS
        };
        let five = markers(&[(0, r(0, 0, 32, 32), r(0, 0, 16, 16), 5)]);
        // Unorm16 halves the colour; levels past the four register-held ones stay in the
        // spill buffer.
        assert_eq!(
            super::plan(&five, 512, 512, unorm16).record_words[0],
            4 + 512 + 1024
        );
    }

    #[test]
    fn list_capacity_counts_only_kept_markers() {
        // The too-deep and capped markers get the skip bit, so coarse lists none of their
        // tiles, whatever depth it sees in each tile.
        let chain: Vec<_> = (0..5)
            .map(|i| {
                let o = 64 * i;
                let region = r(64 + o, 64 + o, 192 + o, 192 + o);
                (i as u32, region, r(96 + o, 96 + o, 160 + o, 160 + o), 0)
            })
            .chain([(9, r(0, 448, 64, 512), r(0, 448, 16, 464), 9)])
            .collect();
        let plan = plan(&markers(&chain), 512, 512);
        assert_eq!(plan.list_capacity, 4 * 64);
        assert_eq!(plan.left_out[4], Some(GlassLeftOut::PhaseCap));
        assert_eq!(plan.left_out[5], None);
        assert_eq!(plan.left_out_offsets, [12, 15]);
    }

    #[test]
    fn atlas_shrinks_after_sustained_low_use() {
        let mut sizer = AtlasSizer::default();
        assert_eq!(sizer.size((1024, 512)), (1024, 512));
        // A frame needing more grows at once, in each dimension.
        assert_eq!(sizer.size((512, 1024)), (1024, 1024));
        for frame in 1..120 {
            let need = if frame % 2 == 0 {
                (512, 256)
            } else {
                (256, 512)
            };
            assert_eq!(sizer.size(need), (1024, 1024), "frame {frame}");
        }
        // The 120th low frame shrinks to the smallest size that fits the whole run.
        assert_eq!(sizer.size((512, 128)), (512, 512));
        assert_eq!(sizer.size((512, 512)), (512, 512));
    }

    #[test]
    fn oversized_atlas_frame_leaves_the_sizer_unchanged() {
        let mut sizer = AtlasSizer::default();
        assert_eq!(sizer.fit((512, 512), 8192), Ok((512, 512)));
        assert_eq!(sizer.fit((512, 32768), 8192), Err((512, 32768)));
        // The next frame that fits gets the atlas it had, not the oversized one.
        assert_eq!(sizer.fit((512, 256), 8192), Ok((512, 512)));
        assert_eq!(sizer.fit((1024, 512), 8192), Ok((1024, 512)));
    }

    #[test]
    fn atlas_does_not_shrink_when_use_alternates() {
        let mut sizer = AtlasSizer::default();
        for frame in 0..1000 {
            let need = if frame % 100 == 0 {
                (1024, 1024)
            } else {
                (512, 256)
            };
            assert_eq!(sizer.size(need), (1024, 1024), "frame {frame}");
        }
        // Just over a quarter of the area is not low use.
        let mut sizer = AtlasSizer::default();
        sizer.size((1024, 1024));
        for _ in 0..200 {
            assert_eq!(sizer.size((1024, 257)), (1024, 1024));
        }
    }

    #[test]
    fn placements_are_grouped_by_phase_in_paint_order() {
        let m = markers(&[
            (0, r(0, 0, 64, 64), r(16, 16, 48, 48), 0),
            (1, r(32, 32, 96, 96), r(48, 48, 80, 80), 0),
            (2, r(256, 0, 320, 64), r(272, 16, 304, 48), 0),
        ]);
        let plan = plan(&m, 512, 512);
        let ids: Vec<_> = plan.placements.iter().map(|p| (p.phase, p.id)).collect();
        assert_eq!(ids, [(1, 0), (1, 2), (2, 1)]);
        assert_eq!(plan.phases, 2);
        let phase_two: Vec<_> = phase_placements(&plan.placements, 2)
            .iter()
            .map(|p| p.id)
            .collect();
        assert_eq!(phase_two, [1]);
        assert_eq!(plan.rows[1].phase, 2);
    }
}
