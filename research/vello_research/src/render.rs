// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Take an encoded scene and create a graph to render it

use crate::recording::{BufferProxy, Command, ImageFormat, ImageProxy, Recording, ResourceProxy};
use crate::shaders::{FullShaders, GlassShaders};
use crate::{AaConfig, RenderParams};

#[cfg(feature = "wgpu")]
use crate::Scene;
use crate::glass::{GlassMarkerRow, MAX_GLASS_PHASES};

use vello_encoding::{
    Encoding, GLASS_MARKER_SKIP, Resolver, VibrantMatrix, WorkgroupSize, make_mask_lut,
    make_mask_lut_16,
};

#[derive(Clone, Copy, Debug)]
enum AtlasProxyAction {
    Created,
    Reused,
    Resized,
}

/// State for a render in progress.
pub struct Render {
    fine_wg_count: Option<WorkgroupSize>,
    fine_resources: Option<FineResources>,
    mask_buf: Option<ResourceProxy>,
    /// A persistent one-identity vibrant table, bound when the scene has no vibrant layers.
    vibrant_identity: Option<BufferProxy>,

    #[cfg(feature = "debug_layers")]
    captured_buffers: Option<CapturedBuffers>,
}

#[cfg(feature = "debug_layers")]
impl Drop for Render {
    fn drop(&mut self) {
        if self.captured_buffers.is_some() {
            unreachable!("Render captured buffers without freeing them");
        }
    }
}

/// Resources produced by pipeline, needed for fine rasterization.
struct FineResources {
    aa_config: AaConfig,

    config_buf: ResourceProxy,
    bump_buf: ResourceProxy,
    tile_buf: ResourceProxy,
    segments_buf: ResourceProxy,
    ptcl_buf: ResourceProxy,
    gradient_image: ResourceProxy,
    info_bin_data_buf: ResourceProxy,
    image_atlas: ResourceProxy,
    blend_spill_buf: ResourceProxy,
    /// The vibrant matrix table, and whether this render uploaded it.
    vibrant_buf: ResourceProxy,
    vibrant_uploaded: bool,

    out_image: ImageProxy,
}

/// A collection of internal buffers that are used for debug visualization when the
/// `debug_layers` feature is enabled. The contents of these buffers remain GPU resident
/// and must be freed directly by the caller.
///
/// Some of these buffers are also scheduled for a download to allow their contents to be
/// processed for CPU-side validation. These buffers are documented as such.
#[cfg(feature = "debug_layers")]
pub struct CapturedBuffers {
    pub sizes: vello_encoding::BufferSizes,

    /// Buffers that remain GPU-only
    pub path_bboxes: BufferProxy,

    /// Buffers scheduled for download
    pub lines: BufferProxy,
}

#[cfg(feature = "debug_layers")]
impl CapturedBuffers {
    pub fn release_buffers(self, recording: &mut Recording) {
        recording.free_buffer(self.path_bboxes);
        recording.free_buffer(self.lines);
    }
}

#[cfg(feature = "wgpu")]
pub(crate) fn render_full(
    scene: &Scene,
    resolver: &mut Resolver,
    shaders: &FullShaders,
    image_atlas: &mut Option<ImageProxy>,
    params: &RenderParams,
    vibrant_identity: BufferProxy,
) -> (Recording, ResourceProxy) {
    render_encoding_full(
        scene.encoding(),
        resolver,
        shaders,
        image_atlas,
        params,
        vibrant_identity,
    )
}

/// Size of `glass_state`'s header, which is zeroed each frame: the running phase, list 0's
/// length, the count of tiles past capacity, the lite pass's test count of commands and the
/// stop count of each of the four phases, as `u32`s.
const GLASS_STATE_HEADER_SIZE: u64 = 32;

/// Words in a resume record's header: command index, clip depth, tile index and marker.
#[cfg(feature = "wgpu")]
const GLASS_RECORD_HEADER_WORDS: u32 = 4;

/// Words of a stopped tile's colour: 256 pixels of four `f32`, or of two packed `unorm16` pairs.
#[cfg(feature = "wgpu")]
const GLASS_COLOR_WORDS: [u32; 2] = [1024, 512];

/// Register-held blend stack levels fine saves at a stop: `BLEND_STACK_SPLIT` in `config.wgsl`.
#[cfg(feature = "wgpu")]
const GLASS_STACK_LEVELS: u32 = 4;

/// Words of one saved blend stack level: one packed colour per pixel of a 16×16 tile.
#[cfg(feature = "wgpu")]
const GLASS_STACK_LEVEL_WORDS: u32 = 256;

/// Resources the start and resume permutations bind after stock fine's: the marker table,
/// the glass state, the backdrop atlas and the glass atlas.
const GLASS_FINE_RESOURCES: usize = 4;

/// Bytes of one phase's indirect dispatch arguments: `(count_x, 1, 1, pad)`.
const GLASS_ARGS_SIZE: u64 = 16;

/// Returns the size in words of a resume record for a phase whose markers sit under at most
/// `depth` open layers, with `resume_color` 0 for `F32` and 1 for `Unorm16`.
///
/// Every record keeps the tile's colour; only the blend stack shrinks with the depth.
#[cfg(feature = "wgpu")]
pub(crate) fn glass_record_words(resume_color: u32, depth: u32) -> u32 {
    GLASS_RECORD_HEADER_WORDS
        + GLASS_COLOR_WORDS[resume_color as usize]
        + depth.min(GLASS_STACK_LEVELS) * GLASS_STACK_LEVEL_WORDS
}

/// Returns the size in bytes of `glass_state` for list 0's and the phases' capacities, with
/// each phase's records `record_words` long.
///
/// The layout, in `u32` words, mirrors `fine.wgsl`: the 8-word header, then list 0
/// (`list_capacity` words), then each phase's tile list (`capacity` words), then each
/// phase's resume records (`capacity` records).
pub(crate) fn glass_state_size(
    list_capacity: u32,
    capacities: [u32; 4],
    record_words: [u32; 4],
) -> u64 {
    let per_phase: u64 = capacities
        .iter()
        .zip(record_words)
        .map(|(&c, words)| u64::from(c) * (1 + u64::from(words)))
        .sum();
    GLASS_STATE_HEADER_SIZE + 4 * u64::from(list_capacity) + 4 * per_phase
}

/// Sets [`GLASS_MARKER_SKIP`] and an empty composite in each left-out `GLASS` object of the
/// packed scene, whose draw data starts at word `draw_data_base`.
///
/// The resolver copies the glass words unchanged, at their offsets in the encoding.
fn skip_glass_markers(packed: &mut [u8], draw_data_base: u32, left_out: &[usize]) {
    for &offset in left_out {
        let at = 4 * (draw_data_base as usize + offset);
        let words = &mut packed[at..at + 12];
        let index = u32::from_ne_bytes(words[..4].try_into().unwrap());
        words[..4].copy_from_slice(&(index | GLASS_MARKER_SKIP).to_ne_bytes());
        words[4..].fill(0);
    }
}

/// One frame's glass settings and resources.
pub(crate) struct GlassRender<'a> {
    pub(crate) shaders: &'a GlassShaders,
    pub(crate) rows: &'a [GlassMarkerRow],
    pub(crate) phases: u32,
    pub(crate) resume_color: u32,
    pub(crate) flags: u32,
    /// Most tiles with glass commands, the capacity of list 0.
    pub(crate) list_capacity: u32,
    pub(crate) capacities: [u32; 4],
    /// Size of each phase's resume records, in words.
    pub(crate) record_words: [u32; 4],
    /// Draw data offsets of the `GLASS` objects the frame leaves out, whose words packing
    /// rewrites so coarse skips them.
    pub(crate) left_out: &'a [usize],
    /// Whether phase 0 runs the start pass and the later phases run; off only in tests.
    pub(crate) start_pass: bool,
    pub(crate) backdrop_atlas: ImageProxy,
    pub(crate) glass_atlas: ImageProxy,
}

/// Create a single recording with coarse, glass phase 0 fine, and the external glass phases.
#[cfg(feature = "wgpu")]
pub(crate) fn render_full_glass(
    scene: &Scene,
    resolver: &mut Resolver,
    shaders: &FullShaders,
    image_atlas: &mut Option<ImageProxy>,
    params: &RenderParams,
    glass: &GlassRender<'_>,
    vibrant_identity: BufferProxy,
) -> (Recording, ResourceProxy) {
    let mut render = Render::with_vibrant_identity(vibrant_identity);
    let mut recording = render.render_encoding_coarse_inner(
        scene.encoding(),
        resolver,
        shaders,
        image_atlas,
        params,
        false,
        Some(glass),
    );
    let out_image = render.out_image();
    render.record_fine_inner(shaders, &mut recording, Some(glass));
    (recording, out_image.into())
}

#[cfg(feature = "wgpu")]
/// Create a single recording with both coarse and fine render stages.
///
/// This function is not recommended when the scene can be complex, as it does not
/// implement robust dynamic memory.
pub(crate) fn render_encoding_full(
    encoding: &Encoding,
    resolver: &mut Resolver,
    shaders: &FullShaders,
    image_atlas: &mut Option<ImageProxy>,
    params: &RenderParams,
    vibrant_identity: BufferProxy,
) -> (Recording, ResourceProxy) {
    let mut render = Render::with_vibrant_identity(vibrant_identity);
    let mut recording =
        render.render_encoding_coarse(encoding, resolver, shaders, image_atlas, params, false);
    let out_image = render.out_image();
    render.record_fine(shaders, &mut recording);
    (recording, out_image.into())
}

impl Default for Render {
    fn default() -> Self {
        Self::new()
    }
}

impl Render {
    pub fn new() -> Self {
        Self {
            fine_wg_count: None,
            fine_resources: None,
            mask_buf: None,
            vibrant_identity: None,
            #[cfg(feature = "debug_layers")]
            captured_buffers: None,
        }
    }

    /// Creates a render that binds `identity` as the vibrant table of scenes without vibrant
    /// layers.
    ///
    /// `identity` must hold [`VibrantMatrix::IDENTITY`](vello_encoding::VibrantMatrix::IDENTITY)
    /// in its GPU layout, and the caller binds it as an external buffer. A render without
    /// one uploads an identity table each frame.
    pub fn with_vibrant_identity(identity: BufferProxy) -> Self {
        Self {
            vibrant_identity: Some(identity),
            ..Self::new()
        }
    }

    /// Prepare a recording for the coarse rasterization phase.
    ///
    /// The `robust` parameter controls whether we're preparing for readback
    /// of the atomic bump buffer, for robust dynamic memory.
    pub fn render_encoding_coarse(
        &mut self,
        encoding: &Encoding,
        resolver: &mut Resolver,
        shaders: &FullShaders,
        persistent_image_atlas: &mut Option<ImageProxy>,
        params: &RenderParams,
        robust: bool,
    ) -> Recording {
        self.render_encoding_coarse_inner(
            encoding,
            resolver,
            shaders,
            persistent_image_atlas,
            params,
            robust,
            None,
        )
    }

    fn render_encoding_coarse_inner(
        &mut self,
        encoding: &Encoding,
        resolver: &mut Resolver,
        shaders: &FullShaders,
        persistent_image_atlas: &mut Option<ImageProxy>,
        params: &RenderParams,
        robust: bool,
        glass: Option<&GlassRender<'_>>,
    ) -> Recording {
        use vello_encoding::RenderConfig;
        let mut recording = Recording::default();
        let mut packed = vec![];

        let (layout, ramps, images) = resolver.resolve(encoding, &mut packed);
        if let Some(glass) = glass {
            skip_glass_markers(&mut packed, layout.draw_data_base, glass.left_out);
        }
        let gradient_image = if ramps.height == 0 {
            ResourceProxy::new_image(1, 1, ImageFormat::Rgba8)
        } else {
            let data: &[u8] = bytemuck::cast_slice(ramps.data);
            ResourceProxy::Image(recording.upload_image(
                ramps.width,
                ramps.height,
                ImageFormat::Rgba8,
                data,
            ))
        };
        let atlas_width = images.width.max(1);
        let atlas_height = images.height.max(1);
        let (image_atlas, atlas_proxy_action) = match persistent_image_atlas {
            Some(proxy) if proxy.width == atlas_width && proxy.height == atlas_height => {
                (*proxy, AtlasProxyAction::Reused)
            }
            Some(proxy) => {
                recording.free_image(*proxy);
                let new_proxy = ImageProxy::new(atlas_width, atlas_height, ImageFormat::Rgba8);
                *persistent_image_atlas = Some(new_proxy);
                (new_proxy, AtlasProxyAction::Resized)
            }
            None => {
                let proxy = ImageProxy::new(atlas_width, atlas_height, ImageFormat::Rgba8);
                *persistent_image_atlas = Some(proxy);
                (proxy, AtlasProxyAction::Created)
            }
        };
        if !matches!(atlas_proxy_action, AtlasProxyAction::Reused)
            || !images.images.is_empty()
            || images.evicted != 0
        {
            if images.evicted == 0 {
                log::debug!(
                    "image atlas {}x{}: {:?}, {} upload(s)",
                    atlas_width,
                    atlas_height,
                    atlas_proxy_action,
                    images.images.len()
                );
            } else {
                log::debug!(
                    "image atlas {}x{}: {:?}, {} upload(s), {} eviction(s)",
                    atlas_width,
                    atlas_height,
                    atlas_proxy_action,
                    images.images.len(),
                    images.evicted
                );
            }
        }
        for image in images.images {
            recording.write_image(image_atlas, image.1, image.2, image.0.clone());
        }
        let mut cpu_config =
            RenderConfig::new(&layout, params.width, params.height, &params.base_color);
        if let Some(glass) = glass {
            let gpu = &mut cpu_config.gpu;
            gpu.glass_markers = glass.rows.len() as u32;
            gpu.glass_phases = glass.phases;
            gpu.glass_resume_color = glass.resume_color;
            gpu.glass_flags = glass.flags;
            gpu.glass_capacity_0 = glass.list_capacity;
            [
                gpu.glass_capacity_1,
                gpu.glass_capacity_2,
                gpu.glass_capacity_3,
                gpu.glass_capacity_4,
            ] = glass.capacities;
            [
                gpu.glass_record_words_1,
                gpu.glass_record_words_2,
                gpu.glass_record_words_3,
                gpu.glass_record_words_4,
            ] = glass.record_words;
        }
        // HACK: The coarse workgroup counts is the number of active bins.
        if (cpu_config.workgroup_counts.coarse.0
            * cpu_config.workgroup_counts.coarse.1
            * cpu_config.workgroup_counts.coarse.2)
            > 256
        {
            log::warn!(
                "Trying to paint too large image. {}x{}.\n\
                See https://github.com/linebender/vello/issues/680 for details",
                params.width,
                params.height
            );
        }
        let buffer_sizes = &cpu_config.buffer_sizes;
        let wg_counts = &cpu_config.workgroup_counts;

        if packed.is_empty() {
            // HACK: wgpu doesn't allow empty buffers, so we make sure that the scene buffer we upload
            // can contain at least one array item.
            // The values passed here should never be read, because the scene size in config
            // is zero.
            packed.resize(size_of::<u32>(), u8::MAX);
        }
        let scene_buf = ResourceProxy::Buffer(recording.upload("vello.scene", packed));
        let config_buf = ResourceProxy::Buffer(
            recording.upload_uniform("vello.config", bytemuck::bytes_of(&cpu_config.gpu)),
        );
        let info_bin_data_buf = ResourceProxy::new_buf(
            buffer_sizes.bin_data.size_in_bytes() as u64,
            "vello.info_bin_data_buf",
        );
        let tile_buf =
            ResourceProxy::new_buf(buffer_sizes.tiles.size_in_bytes().into(), "vello.tile_buf");
        let segments_buf = ResourceProxy::new_buf(
            buffer_sizes.segments.size_in_bytes().into(),
            "vello.segments_buf",
        );
        let ptcl_buf =
            ResourceProxy::new_buf(buffer_sizes.ptcl.size_in_bytes().into(), "vello.ptcl_buf");
        let reduced_buf = ResourceProxy::new_buf(
            buffer_sizes.path_reduced.size_in_bytes().into(),
            "vello.reduced_buf",
        );
        // TODO: really only need pathtag_wgs - 1
        recording.dispatch(
            shaders.pathtag_reduce,
            wg_counts.path_reduce,
            [config_buf, scene_buf, reduced_buf],
        );
        let mut pathtag_parent = reduced_buf;
        let mut large_pathtag_bufs = None;
        let use_large_path_scan = wg_counts.use_large_path_scan && !shaders.pathtag_is_cpu;
        if use_large_path_scan {
            let reduced2_buf = ResourceProxy::new_buf(
                buffer_sizes.path_reduced2.size_in_bytes().into(),
                "vello.reduced2_buf",
            );
            recording.dispatch(
                shaders.pathtag_reduce2,
                wg_counts.path_reduce2,
                [reduced_buf, reduced2_buf],
            );
            let reduced_scan_buf = ResourceProxy::new_buf(
                buffer_sizes.path_reduced_scan.size_in_bytes().into(),
                "reduced_scan_buf",
            );
            recording.dispatch(
                shaders.pathtag_scan1,
                wg_counts.path_scan1,
                [reduced_buf, reduced2_buf, reduced_scan_buf],
            );
            pathtag_parent = reduced_scan_buf;
            large_pathtag_bufs = Some((reduced2_buf, reduced_scan_buf));
        }

        let tagmonoid_buf = ResourceProxy::new_buf(
            buffer_sizes.path_monoids.size_in_bytes().into(),
            "vello.tagmonoid_buf",
        );
        let pathtag_scan = if use_large_path_scan {
            shaders.pathtag_scan_large
        } else {
            shaders.pathtag_scan
        };
        recording.dispatch(
            pathtag_scan,
            wg_counts.path_scan,
            [config_buf, scene_buf, pathtag_parent, tagmonoid_buf],
        );
        recording.free_resource(reduced_buf);
        if let Some((reduced2, reduced_scan)) = large_pathtag_bufs {
            recording.free_resource(reduced2);
            recording.free_resource(reduced_scan);
        }
        let path_bbox_buf = ResourceProxy::new_buf(
            buffer_sizes.path_bboxes.size_in_bytes().into(),
            "vello.path_bbox_buf",
        );
        recording.dispatch(
            shaders.bbox_clear,
            wg_counts.bbox_clear,
            [config_buf, path_bbox_buf],
        );
        let bump_buf = BufferProxy::new(
            buffer_sizes.bump_alloc.size_in_bytes().into(),
            "vello.bump_buf",
        );
        recording.clear_all(bump_buf);
        let bump_buf = ResourceProxy::Buffer(bump_buf);
        let lines_buf =
            ResourceProxy::new_buf(buffer_sizes.lines.size_in_bytes().into(), "vello.lines_buf");
        recording.dispatch(
            shaders.flatten,
            wg_counts.flatten,
            [
                config_buf,
                scene_buf,
                tagmonoid_buf,
                path_bbox_buf,
                bump_buf,
                lines_buf,
            ],
        );
        let draw_reduced_buf = ResourceProxy::new_buf(
            buffer_sizes.draw_reduced.size_in_bytes().into(),
            "vello.draw_reduced_buf",
        );
        recording.dispatch(
            shaders.draw_reduce,
            wg_counts.draw_reduce,
            [config_buf, scene_buf, draw_reduced_buf],
        );
        let draw_monoid_buf = ResourceProxy::new_buf(
            buffer_sizes.draw_monoids.size_in_bytes().into(),
            "vello.draw_monoid_buf",
        );
        let clip_inp_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_inps.size_in_bytes().into(),
            "vello.clip_inp_buf",
        );
        recording.dispatch(
            shaders.draw_leaf,
            wg_counts.draw_leaf,
            [
                config_buf,
                scene_buf,
                draw_reduced_buf,
                path_bbox_buf,
                draw_monoid_buf,
                info_bin_data_buf,
                clip_inp_buf,
            ],
        );
        recording.free_resource(draw_reduced_buf);
        let clip_el_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_els.size_in_bytes().into(),
            "vello.clip_el_buf",
        );
        let clip_bic_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_bics.size_in_bytes().into(),
            "vello.clip_bic_buf",
        );
        if wg_counts.clip_reduce.0 > 0 {
            recording.dispatch(
                shaders.clip_reduce,
                wg_counts.clip_reduce,
                [clip_inp_buf, path_bbox_buf, clip_bic_buf, clip_el_buf],
            );
        }
        let clip_bbox_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_bboxes.size_in_bytes().into(),
            "vello.clip_bbox_buf",
        );
        if wg_counts.clip_leaf.0 > 0 {
            recording.dispatch(
                shaders.clip_leaf,
                wg_counts.clip_leaf,
                [
                    config_buf,
                    clip_inp_buf,
                    path_bbox_buf,
                    clip_bic_buf,
                    clip_el_buf,
                    draw_monoid_buf,
                    clip_bbox_buf,
                ],
            );
        }
        recording.free_resource(clip_inp_buf);
        recording.free_resource(clip_bic_buf);
        recording.free_resource(clip_el_buf);
        let draw_bbox_buf = ResourceProxy::new_buf(
            buffer_sizes.draw_bboxes.size_in_bytes().into(),
            "vello.draw_bbox_buf",
        );
        let bin_header_buf = ResourceProxy::new_buf(
            buffer_sizes.bin_headers.size_in_bytes().into(),
            "vello.bin_header_buf",
        );
        recording.dispatch(
            shaders.binning,
            wg_counts.binning,
            [
                config_buf,
                draw_monoid_buf,
                path_bbox_buf,
                clip_bbox_buf,
                draw_bbox_buf,
                bump_buf,
                info_bin_data_buf,
                bin_header_buf,
            ],
        );
        recording.free_resource(draw_monoid_buf);
        recording.free_resource(clip_bbox_buf);
        // Note: this only needs to be rounded up because of the workaround to store the tile_offset
        // in storage rather than workgroup memory.
        let path_buf =
            ResourceProxy::new_buf(buffer_sizes.paths.size_in_bytes().into(), "vello.path_buf");
        recording.dispatch(
            shaders.tile_alloc,
            wg_counts.tile_alloc,
            [
                config_buf,
                scene_buf,
                draw_bbox_buf,
                bump_buf,
                path_buf,
                tile_buf,
            ],
        );
        recording.free_resource(draw_bbox_buf);
        recording.free_resource(tagmonoid_buf);
        let indirect_count_buf = BufferProxy::new(
            buffer_sizes.indirect_count.size_in_bytes().into(),
            "vello.indirect_count",
        );
        recording.dispatch(
            shaders.path_count_setup,
            wg_counts.path_count_setup,
            [bump_buf, indirect_count_buf.into()],
        );
        let seg_counts_buf = ResourceProxy::new_buf(
            buffer_sizes.seg_counts.size_in_bytes().into(),
            "vello.seg_counts_buf",
        );
        recording.dispatch_indirect(
            shaders.path_count,
            indirect_count_buf,
            0,
            [
                config_buf,
                bump_buf,
                lines_buf,
                path_buf,
                tile_buf,
                seg_counts_buf,
            ],
        );
        recording.dispatch(
            shaders.backdrop,
            wg_counts.backdrop,
            [config_buf, bump_buf, path_buf, tile_buf],
        );
        recording.dispatch(
            shaders.coarse,
            wg_counts.coarse,
            [
                config_buf,
                scene_buf,
                draw_monoid_buf,
                bin_header_buf,
                info_bin_data_buf,
                path_buf,
                tile_buf,
                bump_buf,
                ptcl_buf,
            ],
        );
        recording.dispatch(
            shaders.path_tiling_setup,
            wg_counts.path_tiling_setup,
            [bump_buf, indirect_count_buf.into(), ptcl_buf],
        );
        recording.dispatch_indirect(
            shaders.path_tiling,
            indirect_count_buf,
            0,
            [
                bump_buf,
                seg_counts_buf,
                lines_buf,
                path_buf,
                tile_buf,
                segments_buf,
            ],
        );
        recording.free_buffer(indirect_count_buf);
        recording.free_resource(seg_counts_buf);
        recording.free_resource(scene_buf);
        recording.free_resource(draw_monoid_buf);
        recording.free_resource(bin_header_buf);
        recording.free_resource(path_buf);
        let out_image = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
        let blend_spill_buf = BufferProxy::new(
            buffer_sizes.blend_spill.size_in_bytes().into(),
            "vello.blend_spill",
        );
        let (vibrant_buf, vibrant_uploaded) = match self.vibrant_identity {
            Some(identity) if encoding.vibrant.is_empty() => (identity, false),
            _ => {
                let table = if encoding.vibrant.is_empty() {
                    VibrantMatrix::IDENTITY.gpu_words().to_vec()
                } else {
                    encoding.vibrant.gpu_table()
                };
                let buf = recording.upload("vello.vibrant", bytemuck::cast_slice(&table));
                (buf, true)
            }
        };
        self.fine_wg_count = Some(wg_counts.fine);
        self.fine_resources = Some(FineResources {
            aa_config: params.antialiasing_method,
            config_buf,
            bump_buf,
            tile_buf,
            segments_buf,
            ptcl_buf,
            gradient_image,
            info_bin_data_buf,
            blend_spill_buf: ResourceProxy::Buffer(blend_spill_buf),
            vibrant_buf: ResourceProxy::Buffer(vibrant_buf),
            vibrant_uploaded,
            image_atlas: ResourceProxy::Image(image_atlas),
            out_image,
        });
        if robust {
            recording.download(*bump_buf.as_buf().unwrap());
        }
        recording.free_resource(bump_buf);

        #[cfg(feature = "debug_layers")]
        {
            if robust {
                let path_bboxes = *path_bbox_buf.as_buf().unwrap();
                let lines = *lines_buf.as_buf().unwrap();
                recording.download(lines);

                self.captured_buffers = Some(CapturedBuffers {
                    sizes: cpu_config.buffer_sizes,
                    path_bboxes,
                    lines,
                });
            } else {
                recording.free_resource(path_bbox_buf);
                recording.free_resource(lines_buf);
            }
        }
        #[cfg(not(feature = "debug_layers"))]
        {
            recording.free_resource(path_bbox_buf);
            recording.free_resource(lines_buf);
        }

        recording
    }

    /// Run fine rasterization assuming the coarse phase succeeded.
    pub fn record_fine(&mut self, shaders: &FullShaders, recording: &mut Recording) {
        self.record_fine_inner(shaders, recording, None);
    }

    /// Run fine rasterization, as glass phase 0 followed by the external phases if `glass` is set.
    ///
    /// Glass phase 0 runs the lite permutation over every tile, which lists the tiles with
    /// glass commands and leaves them unwritten, then the start permutation over that list.
    /// Each external phase is followed by a dispatch that resumes the tiles stopped for it.
    fn record_fine_inner(
        &mut self,
        shaders: &FullShaders,
        recording: &mut Recording,
        glass: Option<&GlassRender<'_>>,
    ) {
        let fine_wg_count = self.fine_wg_count.take().unwrap();
        let fine = self.fine_resources.take().unwrap();
        let glass_bufs = glass.map(|glass| {
            let markers = recording.upload("vello.glass_markers", bytemuck::cast_slice(glass.rows));
            let state = BufferProxy::new(
                glass_state_size(glass.list_capacity, glass.capacities, glass.record_words),
                "vello.glass_state",
            );
            recording.push(Command::Clear(state, 0, Some(GLASS_STATE_HEADER_SIZE)));
            (markers, state)
        });
        let glass_resources = glass.zip(glass_bufs).map(|(glass, (markers, state))| {
            [
                ResourceProxy::Buffer(markers),
                ResourceProxy::Buffer(state),
                ResourceProxy::Image(glass.backdrop_atlas),
                ResourceProxy::Image(glass.glass_atlas),
            ]
        });
        let glass_resources = glass_resources.iter().flatten().copied();
        // The shader for every tile and, with glass, the start and resume shaders.
        let (fine_shader, glass_fine, resources) = match fine.aa_config {
            AaConfig::Area => {
                let (shader, glass_fine) = if let Some(glass) = glass {
                    let s = glass.shaders;
                    (s.fine_area_lite, s.fine_area_start.zip(s.fine_area_resume))
                } else {
                    (shaders.fine_area, None)
                };
                let resources = [
                    fine.config_buf,
                    fine.segments_buf,
                    fine.ptcl_buf,
                    fine.info_bin_data_buf,
                    fine.blend_spill_buf,
                    ResourceProxy::Image(fine.out_image),
                    fine.gradient_image,
                    fine.image_atlas,
                    fine.vibrant_buf,
                ];
                (
                    shader.expect("shaders not configured to support AA mode: area"),
                    glass_fine,
                    resources
                        .into_iter()
                        .chain(glass_resources)
                        .collect::<Vec<_>>(),
                )
            }
            _ => {
                if self.mask_buf.is_none() {
                    let mask_lut = match fine.aa_config {
                        AaConfig::Msaa16 => make_mask_lut_16(),
                        AaConfig::Msaa8 => make_mask_lut(),
                        _ => unreachable!(),
                    };
                    let buf = recording.upload("vello.mask_lut", mask_lut);
                    self.mask_buf = Some(buf.into());
                }
                let (fine_shader, glass_fine, mode) = match (fine.aa_config, glass) {
                    (AaConfig::Msaa16, None) => (shaders.fine_msaa16, None, "msaa16"),
                    (AaConfig::Msaa8, None) => (shaders.fine_msaa8, None, "msaa8"),
                    (AaConfig::Msaa16, Some(glass)) => {
                        let s = glass.shaders;
                        (
                            s.fine_msaa16_lite,
                            s.fine_msaa16_start.zip(s.fine_msaa16_resume),
                            "msaa16",
                        )
                    }
                    (AaConfig::Msaa8, Some(glass)) => {
                        let s = glass.shaders;
                        (
                            s.fine_msaa8_lite,
                            s.fine_msaa8_start.zip(s.fine_msaa8_resume),
                            "msaa8",
                        )
                    }
                    _ => unreachable!(),
                };
                let fine_shader = fine_shader
                    .unwrap_or_else(|| panic!("shaders not configured to support AA mode: {mode}"));
                let resources = [
                    fine.config_buf,
                    fine.segments_buf,
                    fine.ptcl_buf,
                    fine.info_bin_data_buf,
                    fine.blend_spill_buf,
                    ResourceProxy::Image(fine.out_image),
                    fine.gradient_image,
                    fine.image_atlas,
                    self.mask_buf.unwrap(),
                    fine.vibrant_buf,
                ];
                (
                    fine_shader,
                    glass_fine,
                    resources
                        .into_iter()
                        .chain(glass_resources)
                        .collect::<Vec<_>>(),
                )
            }
        };
        let Some((glass, (markers, state))) = glass.zip(glass_bufs) else {
            recording.dispatch(fine_shader, fine_wg_count, resources.iter().copied());
            self.free_fine(recording, fine);
            return;
        };
        // The lite pass binds stock fine's resources and the vibrant table, then the glass
        // state.
        let stock_count = resources.len() - GLASS_FINE_RESOURCES;
        let lite_resources = resources[..stock_count]
            .iter()
            .copied()
            .chain([ResourceProxy::Buffer(state)]);
        recording.dispatch(fine_shader, fine_wg_count, lite_resources);
        let (start_shader, resume_shader) =
            glass_fine.expect("lite permutations come with start and resume ones");
        if glass.start_pass {
            // `(count_x, 1, 1, pad)` per phase, written by `glass_resume_setup`, then list
            // 0's, written by `glass_start_setup`.
            let args = BufferProxy::new(
                u64::from(MAX_GLASS_PHASES + 1) * GLASS_ARGS_SIZE,
                "vello.glass_args",
            );
            let setup_resources = [
                fine.config_buf,
                ResourceProxy::Buffer(state),
                ResourceProxy::Buffer(args),
            ];
            recording.dispatch(glass.shaders.glass_start_setup, (1, 1, 1), setup_resources);
            recording.dispatch_indirect(
                start_shader,
                args,
                u64::from(MAX_GLASS_PHASES) * GLASS_ARGS_SIZE,
                resources.iter().copied(),
            );
            for phase in 1..=glass.phases {
                recording.push(Command::External(phase));
                recording.dispatch(glass.shaders.glass_resume_setup, (1, 1, 1), setup_resources);
                recording.dispatch_indirect(
                    resume_shader,
                    args,
                    u64::from(phase - 1) * GLASS_ARGS_SIZE,
                    resources.iter().copied(),
                );
            }
            recording.free_buffer(args);
        }
        recording.free_buffer(markers);
        recording.free_buffer(state);
        self.free_fine(recording, fine);
    }

    /// Frees fine's resources after its dispatches.
    fn free_fine(&mut self, recording: &mut Recording, fine: FineResources) {
        recording.free_resource(fine.config_buf);
        recording.free_resource(fine.tile_buf);
        recording.free_resource(fine.segments_buf);
        recording.free_resource(fine.ptcl_buf);
        recording.free_resource(fine.gradient_image);
        recording.free_resource(fine.info_bin_data_buf);
        recording.free_resource(fine.blend_spill_buf);
        if fine.vibrant_uploaded {
            recording.free_resource(fine.vibrant_buf);
        }
        // TODO: make mask buf persistent
        if let Some(mask_buf) = self.mask_buf.take() {
            recording.free_resource(mask_buf);
        }
    }

    /// Get the output image.
    ///
    /// This is going away, as the caller will add the output image to the bind
    /// map.
    pub fn out_image(&self) -> ImageProxy {
        self.fine_resources.as_ref().unwrap().out_image
    }

    pub fn bump_buf(&self) -> BufferProxy {
        *self
            .fine_resources
            .as_ref()
            .unwrap()
            .bump_buf
            .as_buf()
            .unwrap()
    }

    #[cfg(feature = "debug_layers")]
    pub fn take_captured_buffers(&mut self) -> Option<CapturedBuffers> {
        self.captured_buffers.take()
    }
}
