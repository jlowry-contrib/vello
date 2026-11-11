// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello is a compute-centric 2D graphics rendering engine written in Rust,
//! using [`wgpu`].
//! It efficiently draws large 2D scenes with interactive or near-interactive
//! performance.
//!
//! ![image](https://github.com/linebender/vello/assets/8573618/cc2b742e-2135-4b70-8051-c49aeddb5d19)
//!
//!
//! ## Motivation
//!
//! Vello is meant to fill the same place in the graphics stack as other vector graphics renderers like [Skia](https://skia.org/), [Cairo](https://www.cairographics.org/), and its predecessor project [Piet](https://www.cairographics.org/).
//! On a basic level, that means it provides tools to render shapes, images, gradients, texts, etc, using a PostScript-inspired API, the same that powers SVG files and [the browser `<canvas>` element](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D).
//!
//! Vello's selling point is that it gets better performance than other renderers by better leveraging the GPU.
//! In traditional PostScript-style renderers, some steps of the render process like sorting and clipping either need to be handled in the CPU or done through the use of intermediary textures.
//! Vello avoids this by using prefix-scan algorithms to parallelize work that usually needs to happen in sequence, so that work can be offloaded to the GPU with minimal use of temporary buffers.
//!
//! This means that Vello needs a GPU with support for compute shaders to run.
//!
//!
//! ## Getting started
//!
//! Vello is meant to be integrated deep in UI render stacks.
//! While drawing in a Vello [`Scene`] is easy, actually rendering that scene to a surface setting up a wgpu context, which is a non-trivial task.
//!
//! To use Vello as the renderer for your PDF reader / GUI toolkit / etc, your code will have to look roughly like this:
//!
//! ```ignore
//! // Initialize wgpu and get handles
//! let (width, height) = ...;
//! let device: wgpu::Device = ...;
//! let queue: wgpu::Queue = ...;
//! let mut renderer = Renderer::new(
//!    &device,
//!    RendererOptions {
//!       use_cpu: false,
//!       antialiasing_support: vello::AaSupport::all(),
//!       num_init_threads: NonZeroUsize::new(1),
//!    },
//! ).expect("Failed to create renderer");
//!
//! // Create scene and draw stuff in it
//! let mut scene = vello::Scene::new();
//! scene.fill(
//!    vello::peniko::Fill::NonZero,
//!    vello::Affine::IDENTITY,
//!    vello::Color::from_rgb8(242, 140, 168),
//!    None,
//!    &vello::Circle::new((420.0, 200.0), 120.0),
//! );
//!
//! // Draw more stuff
//! scene.push_layer(...);
//! scene.fill(...);
//! scene.stroke(...);
//! scene.pop_layer(...);
//!
//! let texture = device.create_texture(&...);
//! // Render to a wgpu Texture
//! renderer
//!    .render_to_texture(
//!       &device,
//!       &queue,
//!       &scene,
//!       &texture,
//!       &vello::RenderParams {
//!          base_color: palette::css::BLACK, // Background color
//!          width,
//!          height,
//!          antialiasing_method: AaConfig::Msaa16,
//!       },
//!    )
//!    .expect("Failed to render to a texture");
//! // Do things with surface texture, such as blitting it to the Surface using
//! // wgpu::util::TextureBlitter.
//! ```
//!
//! See the [`research/examples/`](https://github.com/linebender/vello/tree/main/research/examples) folder to see how that code integrates with frameworks like winit.

// LINEBENDER LINT SET - lib.rs - v2
// See https://linebender.org/wiki/canonical-lints/
// These lints aren't included in Cargo.toml because they
// shouldn't apply to examples and tests
#![warn(unused_crate_dependencies)]
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
// Need to allow instead of expect until Rust 1.83 https://github.com/rust-lang/rust/pull/130025
#![allow(missing_docs, reason = "We have many as-yet undocumented items.")]
#![expect(
    missing_debug_implementations,
    clippy::cast_possible_truncation,
    clippy::missing_assert_message,
    reason = "Deferred"
)]
#![allow(
    clippy::todo,
    unreachable_pub,
    unnameable_types,
    reason = "Deferred, only apply in some feature sets so not expect"
)]

mod debug;
pub mod glass;
mod recording;
mod render;
mod scene;
mod shaders;

#[cfg(feature = "wgpu")]
pub mod util;
#[cfg(feature = "wgpu")]
mod wgpu_engine;

pub mod low_level {
    //! Utilities which can be used to create an alternative Vello renderer to [`Renderer`][crate::Renderer].
    //!
    //! These APIs have not been carefully designed, and might not be powerful enough for this use case.

    pub use crate::debug::DebugLayers;
    pub use crate::recording::{
        BindType, BufferProxy, Command, ImageFormat, ImageProxy, Recording, ResourceId,
        ResourceProxy, ShaderId,
    };
    pub use crate::render::Render;
    pub use crate::shaders::FullShaders;

    pub use vello_encoding;
}
/// Styling and composition primitives.
pub use peniko;
/// 2D geometry, with a focus on curves.
pub use peniko::kurbo;

#[cfg(feature = "wgpu")]
use peniko::ImageData;
#[cfg(feature = "wgpu")]
pub use wgpu;

pub use scene::{DrawGlyphs, Scene};
pub use vello_encoding::{
    COMPOSE_PLUS_DARKER, FontEmbolden, Glyph, MAX_VIBRANT_MATRICES, NormalizedCoord, VibrantMatrix,
};

use low_level::ShaderId;
#[cfg(feature = "wgpu")]
use low_level::{FullShaders, Recording, Render};
use thiserror::Error;

#[cfg(feature = "wgpu")]
use debug::DebugLayers;
#[cfg(feature = "wgpu")]
use vello_encoding::{BumpAllocators, Resolver};
#[cfg(feature = "wgpu")]
use wgpu_engine::{ExternalPasses, ExternalResource, WgpuEngine};

#[cfg(feature = "wgpu")]
use glass::{GlassAtlasFormat, GlassFrame, GlassPasses, GlassResumeFormat};
#[cfg(feature = "wgpu")]
use std::{num::NonZeroUsize, sync::atomic::AtomicBool};
#[cfg(feature = "wgpu")]
use wgpu::{Device, Queue, TextureView};
#[cfg(all(feature = "wgpu", feature = "wgpu-profiler"))]
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};

/// Represents the anti-aliasing method to use during a render pass.
///
/// Can be configured for a render operation by setting [`RenderParams::antialiasing_method`].
/// Each value of this can only be used if the corresponding field on [`AaSupport`] was used.
///
/// This can be converted into an `AaSupport` using [`Iterator::collect`],
/// as `AaSupport` implements `FromIterator`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AaConfig {
    /// Area anti-aliasing, where the alpha value for a pixel is computed from integrating
    /// the winding number over its square area.
    ///
    /// This technique produces very accurate values when the shape has winding number of 0 or 1
    /// everywhere, but can result in conflation artifacts otherwise.
    /// It generally has better performance than the multi-sampling methods.
    ///
    /// Can only be used if [enabled][AaSupport::area] for the `Renderer`.
    Area,
    /// 8x Multisampling
    ///
    /// Can only be used if [enabled][AaSupport::msaa8] for the `Renderer`.
    Msaa8,
    /// 16x Multisampling
    ///
    /// Can only be used if [enabled][AaSupport::msaa16] for the `Renderer`.
    Msaa16,
}

/// Represents the set of anti-aliasing configurations to enable during pipeline creation.
///
/// This is configured at `Renderer` creation time ([`Renderer::new`]) by setting
/// [`RendererOptions::antialiasing_support`].
///
/// This can be created from a set of `AaConfig` using [`Iterator::collect`],
/// as `AaSupport` implements `FromIterator`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AaSupport {
    /// Support [`AaConfig::Area`].
    pub area: bool,
    /// Support [`AaConfig::Msaa8`].
    pub msaa8: bool,
    /// Support [`AaConfig::Msaa16`].
    pub msaa16: bool,
}

impl AaSupport {
    /// Support every anti-aliasing method.
    ///
    /// This might increase startup time, as more shader variations must be compiled.
    pub fn all() -> Self {
        Self {
            area: true,
            msaa8: true,
            msaa16: true,
        }
    }

    /// Support only [`AaConfig::Area`].
    ///
    /// This should be the default choice for most users.
    pub fn area_only() -> Self {
        Self {
            area: true,
            msaa8: false,
            msaa16: false,
        }
    }
}

impl FromIterator<AaConfig> for AaSupport {
    fn from_iter<T: IntoIterator<Item = AaConfig>>(iter: T) -> Self {
        let mut result = Self {
            area: false,
            msaa8: false,
            msaa16: false,
        };
        for config in iter {
            match config {
                AaConfig::Area => result.area = true,
                AaConfig::Msaa8 => result.msaa8 = true,
                AaConfig::Msaa16 => result.msaa16 = true,
            }
        }
        result
    }
}

/// Errors that can occur in Vello.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// There is no available device with the features required by Vello.
    #[cfg(feature = "wgpu")]
    #[error("Couldn't find suitable device")]
    NoCompatibleDevice,
    /// Failed to create surface.
    /// See [`wgpu::CreateSurfaceError`] for more information.
    #[cfg(feature = "wgpu")]
    #[error("Couldn't create wgpu surface")]
    WgpuCreateSurfaceError(#[from] wgpu::CreateSurfaceError),
    /// Surface doesn't support the required texture formats.
    /// Make sure that you have a surface which provides one of
    /// [`TextureFormat::Rgba8Unorm`][wgpu::TextureFormat::Rgba8Unorm]
    /// or [`TextureFormat::Bgra8Unorm`][wgpu::TextureFormat::Bgra8Unorm] as texture formats.
    // TODO: Why does this restriction exist?
    #[cfg(feature = "wgpu")]
    #[error("Couldn't find `Rgba8Unorm` or `Bgra8Unorm` texture formats for surface")]
    UnsupportedSurfaceFormat,

    /// Used a buffer inside a recording while it was not available.
    /// Check if you have created it and not freed before its last usage.
    #[cfg(feature = "wgpu")]
    #[error("Buffer '{0}' is not available but used for {1}")]
    UnavailableBufferUsed(&'static str, &'static str),
    /// Failed to async map a buffer.
    /// See [`wgpu::BufferAsyncError`] for more information.
    #[cfg(feature = "wgpu")]
    #[error("Failed to async map a buffer")]
    BufferAsyncError(#[from] wgpu::BufferAsyncError),
    /// Failed to download an internal buffer for debug visualization.
    #[cfg(feature = "wgpu")]
    #[cfg(feature = "debug_layers")]
    #[error("Failed to download internal buffer '{0}' for visualization")]
    DownloadError(&'static str),

    #[cfg(feature = "wgpu")]
    #[error("wgpu Error from scope")]
    WgpuErrorFromScope(#[from] wgpu::Error),

    /// A glass atlas needs a texture larger than the device allows.
    /// The fields are the atlas width and height and the device's limit, in pixels.
    #[cfg(feature = "wgpu")]
    #[error("Glass atlas of {0}x{1} exceeds the device's {2} px texture limit")]
    GlassAtlasTooLarge(u32, u32, u32),

    /// Failed to create [`GpuProfiler`].
    /// See [`wgpu_profiler::CreationError`] for more information.
    #[cfg(feature = "wgpu-profiler")]
    #[error("Couldn't create wgpu profiler")]
    #[doc(hidden)] // End-users of Vello should not have `wgpu-profiler` enabled.
    ProfilerCreationError(#[from] wgpu_profiler::CreationError),

    /// Failed to compile the shaders.
    #[cfg(feature = "hot_reload")]
    #[error("Failed to compile shaders:\n{0}")]
    #[doc(hidden)] // End-users of Vello should not have `hot_reload` enabled.
    ShaderCompilation(#[from] vello_shaders::compile::ErrorVec),
}

#[cfg_attr(
    not(feature = "wgpu"),
    expect(dead_code, reason = "this can be unused when wgpu feature is not used")
)]
pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

/// Renders a scene into a texture or surface.
///
/// Currently, each renderer only supports a single surface format, if it
/// supports drawing to surfaces at all.
/// This is an assumption which is known to be limiting, and is planned to change.
#[cfg(feature = "wgpu")]
pub struct Renderer {
    options: RendererOptions,
    engine: WgpuEngine,
    resolver: Resolver,
    image_atlas: Option<recording::ImageProxy>,
    shaders: FullShaders,
    /// Built on the first glass frame, so renderers that draw no glass never compile them.
    glass_shaders: Option<shaders::GlassShaders>,
    glass_backdrop: Option<(wgpu::Texture, TextureView)>,
    glass_backdrop_sizer: glass::AtlasSizer,
    glass_atlas: Option<(wgpu::Texture, TextureView)>,
    glass_atlas_sizer: glass::AtlasSizer,
    glass_resume_format: GlassResumeFormat,
    glass_atlas_format: GlassAtlasFormat,
    /// Markers the glass state's limits left out of the last glass frame, logged on change.
    glass_state_full: usize,
    glass_stops_disabled: bool,
    glass_permutation_forced: bool,
    glass_start_pass_disabled: bool,
    glass_lite_commands_counted: bool,
    glass_state_limit_for_tests: Option<u64>,
    glass_capacity_for_tests: (Option<u32>, Option<u32>),
    /// The vibrant table bound to frames without vibrant layers: one identity matrix, made
    /// once so those frames upload nothing.
    vibrant_identity: (recording::BufferProxy, wgpu::Buffer),
    #[cfg(feature = "debug_layers")]
    debug: debug::DebugRenderer,
    #[cfg(feature = "wgpu-profiler")]
    #[doc(hidden)] // End-users of Vello should not have `wgpu-profiler` enabled.
    /// The profiler used with events for this renderer. This is *not* treated as public API.
    pub profiler: GpuProfiler,
    #[cfg(feature = "wgpu-profiler")]
    #[doc(hidden)] // End-users of Vello should not have `wgpu-profiler` enabled.
    /// The results from profiling. This is *not* treated as public API.
    pub profile_result: Option<Vec<wgpu_profiler::GpuTimerQueryResult>>,
}
// This is not `Send` (or `Sync`) on WebAssembly as the
// underlying wgpu types are not. This can be enabled with the
// `fragile-send-sync-non-atomic-wasm` feature in wgpu.
// See https://github.com/gfx-rs/wgpu/discussions/4127 for
// further discussion of this topic.
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
static_assertions::assert_impl_all!(Renderer: Send);

/// Parameters used in a single render that are configurable by the client.
///
/// These are used in [`Renderer::render_to_texture`].
pub struct RenderParams {
    /// The background color applied to the target. This value is only applicable to the full
    /// pipeline.
    pub base_color: peniko::Color,

    /// Dimensions of the rasterization target
    pub width: u32,
    pub height: u32,

    /// The anti-aliasing algorithm. The selected algorithm must have been initialized while
    /// constructing the `Renderer`.
    pub antialiasing_method: AaConfig,
}

#[cfg(feature = "wgpu")]
/// Options which are set at renderer creation time, used in [`Renderer::new`].
pub struct RendererOptions {
    /// If true, run all stages up to fine rasterization on the CPU.
    ///
    /// This is not a recommended configuration as it is expected to have poor performance,
    /// but it can be useful for debugging.
    // TODO: Consider evolving this so that the CPU stages can be configured dynamically via
    // `RenderParams`.
    pub use_cpu: bool,

    /// Represents the enabled set of AA configurations. This will be used to determine which
    /// pipeline permutations should be compiled at startup.
    ///
    /// By default this will be all modes, to support the widest range of.
    /// It is recommended that most users configure this.
    pub antialiasing_support: AaSupport,

    /// How many threads to use for initialisation of shaders.
    ///
    /// Use `Some(1)` to use a single thread. This is recommended when on macOS
    /// (see <https://github.com/bevyengine/bevy/pull/10812#discussion_r1496138004>)
    ///
    /// Set to `None` to use a heuristic which will use many but not all threads
    ///
    /// Has no effect on WebAssembly
    ///
    /// Will default to `None` on most platforms, `Some(1)` on macOS.
    pub num_init_threads: Option<NonZeroUsize>,

    /// The pipeline cache to use when creating the shaders.
    ///
    /// For much more discussion of expected usage patterns, see the documentation on that type.
    pub pipeline_cache: Option<wgpu::PipelineCache>,
}

#[cfg(feature = "wgpu")]
impl Default for RendererOptions {
    fn default() -> Self {
        Self {
            use_cpu: false,
            antialiasing_support: AaSupport::all(),
            #[cfg(target_os = "macos")]
            num_init_threads: NonZeroUsize::new(1),
            #[cfg(not(target_os = "macos"))]
            num_init_threads: None,
            pipeline_cache: None,
        }
    }
}

/// Sizes a glass atlas for a frame that needs `size`, as `sizer` decides, and returns a
/// proxy for it.
///
/// A format change recreates the atlas at `size`.
#[cfg(feature = "wgpu")]
fn fit_glass_atlas(
    atlas: &mut Option<(wgpu::Texture, TextureView)>,
    sizer: &mut glass::AtlasSizer,
    device: &Device,
    label: &'static str,
    format: recording::ImageFormat,
    size: (u32, u32),
) -> Result<recording::ImageProxy> {
    let wgpu_format = format.to_wgpu();
    let current = atlas
        .as_ref()
        .filter(|(texture, _)| texture.format() == wgpu_format)
        .map(|(texture, _)| (texture.width(), texture.height()));
    if current.is_none() {
        *sizer = glass::AtlasSizer::default();
    }
    let max = device.limits().max_texture_dimension_2d;
    debug_assert!(
        size.0 <= max && size.1 <= max,
        "glass plan sized an atlas of {size:?} past the {max} px texture limit"
    );
    let (width, height) = sizer
        .fit(size, max)
        .map_err(|(width, height)| Error::GlassAtlasTooLarge(width, height, max))?;
    if current != Some((width, height)) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            // Hooks and tests copy into and out of the atlases.
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        *atlas = Some((texture, view));
    }
    Ok(recording::ImageProxy::new(width, height, format))
}

#[cfg(feature = "wgpu")]
struct RenderResult {
    bump: Option<BumpAllocators>,
    #[cfg(feature = "debug_layers")]
    captured: Option<render::CapturedBuffers>,
}

#[cfg(feature = "wgpu")]
impl Renderer {
    /// Creates a new renderer for the specified device.
    pub fn new(device: &Device, options: RendererOptions) -> Result<Self> {
        let mut engine = WgpuEngine::new(options.use_cpu, options.pipeline_cache.clone());
        // If we are running in parallel (i.e. the number of threads is not 1)
        if options.num_init_threads != NonZeroUsize::new(1) {
            #[cfg(not(target_arch = "wasm32"))]
            engine.use_parallel_initialisation();
        }
        let shaders = shaders::full_shaders(device, &mut engine, &options)?;
        #[cfg(not(target_arch = "wasm32"))]
        engine.build_shaders_if_needed(device, options.num_init_threads);
        #[cfg(feature = "debug_layers")]
        let debug = debug::DebugRenderer::new(device, wgpu::TextureFormat::Rgba8Unorm, &mut engine);
        let identity = VibrantMatrix::IDENTITY.gpu_words();
        let identity: &[u8] = bytemuck::cast_slice(&identity);
        let vibrant_identity = (
            recording::BufferProxy::new(identity.len() as u64, "vello.vibrant_identity"),
            wgpu::util::DeviceExt::create_buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("vello.vibrant_identity"),
                    contents: identity,
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ),
        );

        Ok(Self {
            options,
            engine,
            resolver: Resolver::new(),
            image_atlas: None,
            shaders,
            glass_shaders: None,
            glass_backdrop: None,
            glass_backdrop_sizer: glass::AtlasSizer::default(),
            glass_atlas: None,
            glass_atlas_sizer: glass::AtlasSizer::default(),
            glass_resume_format: GlassResumeFormat::default(),
            glass_atlas_format: GlassAtlasFormat::default(),
            glass_state_full: 0,
            glass_stops_disabled: false,
            glass_permutation_forced: false,
            glass_start_pass_disabled: false,
            glass_lite_commands_counted: false,
            glass_state_limit_for_tests: None,
            glass_capacity_for_tests: (None, None),
            vibrant_identity,
            #[cfg(feature = "debug_layers")]
            debug,
            #[cfg(feature = "wgpu-profiler")]
            profiler: GpuProfiler::new(device, GpuProfilerSettings::default())?,
            #[cfg(feature = "wgpu-profiler")]
            profile_result: None,
        })
    }

    /// Renders a scene to the target texture.
    ///
    /// The texture is assumed to be of the specified dimensions and have been created with
    /// the [`wgpu::TextureFormat::Rgba8Unorm`] format and the [`wgpu::TextureUsages::STORAGE_BINDING`]
    /// flag set.
    ///
    /// If you want to render Vello content to a surface (such as in a UI toolkit), you have two options:
    /// 1) Render to an intermediate texture, which is the same size as the surface.
    ///    You would then use [`TextureBlitter`][wgpu::util::TextureBlitter] to blit the rendered result from
    ///    that texture to the surface.
    ///    This pattern is supported by the [`util`] module.
    /// 2) Call `render_to_texture` directly on the [`SurfaceTexture`][wgpu::SurfaceTexture]'s texture, if
    ///    it has the right usages. This should generally be avoided, as some GPUs assume that you will not
    ///    be rendering to the surface using a compute pipeline, and optimise accordingly.
    pub fn render_to_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        texture: &TextureView,
        params: &RenderParams,
    ) -> Result<()> {
        let (recording, target) = render::render_full(
            scene,
            &mut self.resolver,
            &self.shaders,
            &mut self.image_atlas,
            params,
            self.vibrant_identity.0,
        );
        let external_resources = [
            ExternalResource::Image(*target.as_image().unwrap(), texture),
            ExternalResource::Buffer(self.vibrant_identity.0, &self.vibrant_identity.1),
        ];
        self.engine.run_recording(
            device,
            queue,
            &recording,
            &external_resources,
            "render_to_texture",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
            None,
        )?;
        // N.B. This is horrible; this integration of wgpu-profiler really needs some work...
        #[cfg(feature = "wgpu-profiler")]
        {
            self.profiler.end_frame().unwrap();
            if let Some(result) = self
                .profiler
                .process_finished_frame(queue.get_timestamp_period())
            {
                self.profile_result = Some(result);
            }
        }

        Ok(())
    }

    /// Renders a scene with glass to the target texture, calling `passes` once per glass phase.
    ///
    /// The target is as for [`render_to_texture`](Self::render_to_texture). A scene whose
    /// markers all fall outside the target renders as `render_to_texture` does, as does any
    /// scene when the CPU stages are in use (logged as `glass.cpu.unsupported`). The first
    /// frame with markers on the target compiles the glass pipelines.
    ///
    /// # Errors
    ///
    /// Returns the errors of `render_to_texture`. Markers whose atlas slots would pass the
    /// device's texture limit are left out, so [`Error::GlassAtlasTooLarge`] is only a
    /// backstop.
    pub fn render_to_texture_with_glass(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        texture: &TextureView,
        params: &RenderParams,
        passes: &mut dyn GlassPasses,
    ) -> Result<()> {
        let markers = &scene.encoding().glass;
        let device_limits = device.limits();
        let limits = glass::GlassLimits {
            // The shaders index `glass_state` with `u32` words.
            state_size: device_limits
                .max_storage_buffer_binding_size
                .min(device_limits.max_buffer_size)
                .min(4 * u64::from(u32::MAX))
                .min(self.glass_state_limit_for_tests.unwrap_or(u64::MAX)),
            workgroups: device_limits.max_compute_workgroups_per_dimension,
            resume_color: match self.glass_resume_format {
                GlassResumeFormat::F32 => 0,
                GlassResumeFormat::Unorm16 => 1,
            },
            texture_side: device_limits.max_texture_dimension_2d,
        };
        let mut plan = glass::plan(markers, params.width, params.height, limits);
        let state_full = plan.state_full();
        if state_full != self.glass_state_full {
            self.glass_state_full = state_full;
            let (over_size, over_dispatch) = (plan.over_size, plan.over_dispatch);
            let over_atlas = plan.over_atlas;
            let (limit, workgroups) = (limits.state_size, limits.workgroups);
            let texture_side = limits.texture_side;
            if state_full > 0 {
                log::warn!(
                    "glass.state.full left_out={state_full} over_size={over_size} \
                     limit={limit} over_dispatch={over_dispatch} workgroups={workgroups} \
                     over_atlas={over_atlas} texture_side={texture_side}"
                );
            } else {
                log::info!(
                    "glass.state.full left_out=0 limit={limit} workgroups={workgroups} \
                     texture_side={texture_side}"
                );
            }
        }
        if let (Some(list), _) = self.glass_capacity_for_tests {
            plan.list_capacity = plan.list_capacity.min(list);
        }
        if let (_, Some(phase)) = self.glass_capacity_for_tests {
            plan.capacities = plan.capacities.map(|c| c.min(phase));
        }
        if plan.placements.is_empty() {
            if !self.glass_permutation_forced {
                return self.render_to_texture(device, queue, scene, texture, params);
            }
            plan = glass::forced_plan(markers.len());
        }
        if self.engine.use_cpu {
            log::warn!("glass.cpu.unsupported markers={}", plan.placements.len());
            return self.render_to_texture(device, queue, scene, texture, params);
        }
        let counted = self.glass_lite_commands_counted;
        let glass_shaders = match self.glass_shaders.take() {
            Some(glass_shaders) if glass_shaders.lite_counted == counted => glass_shaders,
            _ => shaders::glass_shaders(
                device,
                &mut self.engine,
                self.options.antialiasing_support,
                counted,
            )?,
        };
        let glass_shaders = self.glass_shaders.insert(glass_shaders);
        let glass_format = match self.glass_atlas_format {
            GlassAtlasFormat::Rgba16Float => recording::ImageFormat::Rgba16Float,
            GlassAtlasFormat::Rgba8Unorm => recording::ImageFormat::Rgba8,
        };
        let backdrop_proxy = fit_glass_atlas(
            &mut self.glass_backdrop,
            &mut self.glass_backdrop_sizer,
            device,
            "vello.glass_backdrop",
            recording::ImageFormat::Rgba8,
            plan.backdrop_size,
        )?;
        let glass_proxy = fit_glass_atlas(
            &mut self.glass_atlas,
            &mut self.glass_atlas_sizer,
            device,
            "vello.glass_atlas",
            glass_format,
            plan.glass_size,
        )?;
        passes.prepare(
            device,
            queue,
            &GlassFrame {
                markers: &plan.placements,
                phases: plan.phases,
            },
        );
        let glass = render::GlassRender {
            shaders: glass_shaders,
            rows: &plan.rows,
            phases: plan.phases,
            resume_color: limits.resume_color,
            // Bit 0 skips every stop.
            flags: u32::from(self.glass_stops_disabled),
            list_capacity: plan.list_capacity,
            capacities: plan.capacities,
            record_words: plan.record_words,
            left_out: &plan.left_out_offsets,
            start_pass: !self.glass_start_pass_disabled,
            backdrop_atlas: backdrop_proxy,
            glass_atlas: glass_proxy,
        };
        let (recording, target) = render::render_full_glass(
            scene,
            &mut self.resolver,
            &self.shaders,
            &mut self.image_atlas,
            params,
            &glass,
            self.vibrant_identity.0,
        );
        let (backdrop, backdrop_view) = self.glass_backdrop.as_ref().unwrap();
        let (glass_atlas, glass_view) = self.glass_atlas.as_ref().unwrap();
        let external_resources = [
            ExternalResource::Image(*target.as_image().unwrap(), texture),
            ExternalResource::Image(backdrop_proxy, backdrop_view),
            ExternalResource::Image(glass_proxy, glass_view),
            ExternalResource::Buffer(self.vibrant_identity.0, &self.vibrant_identity.1),
        ];
        let mut external = ExternalPasses {
            passes,
            backdrop,
            glass: glass_atlas,
            placements: &plan.placements,
        };
        self.engine.run_recording(
            device,
            queue,
            &recording,
            &external_resources,
            "render_to_texture_with_glass",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
            Some(&mut external),
        )?;
        #[cfg(feature = "wgpu-profiler")]
        {
            self.profiler.end_frame().unwrap();
            if let Some(result) = self
                .profiler
                .process_finished_frame(queue.get_timestamp_period())
            {
                self.profile_result = Some(result);
            }
        }
        Ok(())
    }

    /// Returns whether the glass fine permutations have been built, for tests.
    #[doc(hidden)]
    pub fn glass_pipelines_compiled(&self) -> bool {
        self.glass_shaders.is_some()
    }

    /// Returns how many shaders the engine holds, for tests.
    #[doc(hidden)]
    pub fn shader_count(&self) -> usize {
        self.engine.shader_count()
    }

    /// Reads back the `glass_state` buffer the last glass frame returned to the pool, for tests.
    ///
    /// Returns an empty vector if no glass frame has run.
    ///
    /// # Panics
    ///
    /// Panics if mapping the read-back buffer fails.
    #[doc(hidden)]
    pub fn read_glass_state(&self, device: &Device, queue: &Queue) -> Vec<u8> {
        let Some(state) = self.engine.pooled_buffer("vello.glass_state") else {
            return Vec::new();
        };
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vello.glass_state_readback"),
            size: state.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vello.glass_state_readback"),
        });
        encoder.copy_buffer_to_buffer(state, 0, &readback, 0, state.size());
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        util::block_on_wgpu(device, receiver.receive())
            .expect("channel was closed")
            .expect("mapping the glass state failed");
        let data = slice.get_mapped_range().unwrap().to_vec();
        readback.unmap();
        data
    }

    /// Sets how fine saves its colour state when a tile stops for glass.
    pub fn set_glass_resume_format(&mut self, format: GlassResumeFormat) {
        self.glass_resume_format = format;
    }

    /// Sets the texel format of the glass atlas handed to the glass passes.
    pub fn set_glass_atlas_format(&mut self, format: GlassAtlasFormat) {
        self.glass_atlas_format = format;
    }

    /// Makes glass frames skip every stop, a control for the split tests.
    #[doc(hidden)]
    pub fn set_glass_stops_disabled(&mut self, disabled: bool) {
        self.glass_stops_disabled = disabled;
    }

    /// Makes frames without placed markers run the glass fine permutation, a control for
    /// the split bench.
    ///
    /// Such a frame, rendered with
    /// [`render_to_texture_with_glass`](Self::render_to_texture_with_glass), runs phase 0
    /// with an empty marker table and no glass phase. `render_to_texture` is unaffected.
    #[doc(hidden)]
    pub fn set_glass_permutation_forced(&mut self, forced: bool) {
        self.glass_permutation_forced = forced;
    }

    /// Makes glass frames run only the lite fine pass, which leaves every tile with glass
    /// commands unwritten, for the split tests.
    #[doc(hidden)]
    pub fn set_glass_start_pass_disabled(&mut self, disabled: bool) {
        self.glass_start_pass_disabled = disabled;
    }

    /// Makes glass frames count the commands the lite fine pass runs, in `glass_state`
    /// word 3, for the split tests.
    ///
    /// The count runs in separate lite permutations, built on the next glass frame.
    #[doc(hidden)]
    pub fn set_glass_lite_commands_counted(&mut self, counted: bool) {
        self.glass_lite_commands_counted = counted;
    }

    /// Lowers the `glass_state` size limit below the device's, for tests.
    #[doc(hidden)]
    pub fn set_glass_state_limit_for_tests(&mut self, limit: Option<u64>) {
        self.glass_state_limit_for_tests = limit;
    }

    /// Caps the capacity of list 0 and of each phase below the plan's, for tests; the setups
    /// then leave the excess out and count it in `glass_state` word 2.
    #[doc(hidden)]
    pub fn set_glass_capacity_for_tests(&mut self, list: Option<u32>, phase: Option<u32>) {
        self.glass_capacity_for_tests = (list, phase);
    }

    /// Overwrite `image` with `texture`.
    ///
    /// Most users should prefer [`register_texture`](Self::register_texture), which
    /// ergonomically encapulates these requirements.
    ///
    /// `texture` must have the [`wgpu::TextureFormat::Rgba8Unorm`] format and
    /// the [`wgpu::TextureUsages::COPY_SRC`] flag set. The `Rgba8UnormSrgb` format
    /// might also be supported, but this is not tested.
    ///
    /// The given `Texture`'s data will be copied into the slot in Vello's image
    /// atlas where the image would be placed each frame.
    /// This has the effect that wherever you request the image to be drawn (e.g.
    /// through [`Scene::draw_image`]), the data from the texture will be used instead.
    ///
    /// Correct behaviour is not guaranteed if the texture does not have the same
    /// dimensions as the image, nor if an image which uses the same [blob] but different
    /// dimensions would be rendered.
    ///
    /// [blob]: peniko::ImageData::data
    pub fn override_image(
        &mut self,
        image: &ImageData,
        texture: Option<wgpu::TexelCopyTextureInfoBase<wgpu::Texture>>,
    ) -> Option<wgpu::TexelCopyTextureInfoBase<wgpu::Texture>> {
        self.resolver.mark_image_dirty(image);
        match texture {
            Some(texture) => self.engine.image_overrides.insert(image.data.id(), texture),
            None => self.engine.image_overrides.remove(&image.data.id()),
        }
    }

    /// Marks `image` as dirty in the atlas cache, so its override texture will be recopied on
    /// the next render that uses it.
    ///
    /// Call this for any `image` whose existing override texture has changed since the last render.
    /// Otherwise, stale image data from the atlas may be used.
    pub fn mark_override_image_dirty(&mut self, image: &ImageData) {
        self.resolver.mark_image_dirty(image);
    }

    /// Register a [`wgpu::Texture`] with Vello, to allow drawing GPU-resident data.
    ///
    /// The returned `Image` can be used in [`Scene`]s (only those rendered with this `Renderer`)
    /// Rendering Scenes which use this `Image` with other `Renderer`s will panic.
    ///  
    /// `texture` must have the [`wgpu::TextureFormat::Rgba8Unorm`] format and
    /// the [`wgpu::TextureUsages::COPY_SRC`] flag set. This is because the data will
    /// be copied into Vello's image atlas at the start of each frame.
    /// The `Rgba8UnormSrgb` format might also be supported, but this is not tested.
    /// The texture is assumed to have unpremultiplied alpha.
    ///
    /// This is a utility wrapper around [`override_image`](Self::override_image).
    /// For greater control, use that method.
    ///
    /// If the texture is no longer active then it should be unregistered using [`unregister_texture`](Self::unregister_texture)
    pub fn register_texture(&mut self, texture: wgpu::Texture) -> ImageData {
        // Create a fake, empty blob which will be used to back the returned image
        // This image data will never be read by Vello, due to being added to
        // image_overrides, below.
        let fake_blob = peniko::Blob::new(std::sync::Arc::new(&[]));

        let image = ImageData {
            data: fake_blob,
            format: peniko::ImageFormat::Rgba8,
            alpha_type: peniko::ImageAlphaType::Alpha,
            width: texture.width(),
            height: texture.height(),
        };

        // For this utility API, we take the full texture and use the base layer and mip level
        let texture_base = wgpu::TexelCopyTextureInfoBase {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        };

        // We overwrite any attempt to use the fake blob, instead reading from the texture.
        self.override_image(&image, Some(texture_base));

        image
    }

    /// Unregister a [`wgpu::Texture`] that was registered with [`register_texture`](Self::register_texture).
    pub fn unregister_texture(&mut self, handle: ImageData) {
        self.override_image(&handle, None);
    }

    /// Reload the shaders. This should only be used during `vello` development
    #[cfg(feature = "hot_reload")]
    #[doc(hidden)] // End-users of Vello should not have `hot_reload` enabled.
    pub async fn reload_shaders(&mut self, device: &Device) -> Result<(), Error> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut engine = WgpuEngine::new(self.options.use_cpu, self.options.pipeline_cache.clone());
        // We choose not to initialise these shaders in parallel, to ensure the error scope works correctly
        let shaders = shaders::full_shaders(device, &mut engine, &self.options)?;
        #[cfg(feature = "debug_layers")]
        let debug = debug::DebugRenderer::new(device, wgpu::TextureFormat::Rgba8Unorm, &mut engine);
        let error = scope.pop().await;
        if let Some(error) = error {
            return Err(error.into());
        }
        self.engine = engine;
        self.image_atlas = None;
        self.shaders = shaders;
        self.glass_shaders = None;
        #[cfg(feature = "debug_layers")]
        {
            self.debug = debug;
        }
        Ok(())
    }

    /// Renders a scene to the target texture using an async pipeline.
    ///
    /// Almost all consumers should prefer [`Self::render_to_texture`].
    ///
    /// The return value is the value of the `BumpAllocators` in this rendering, which is currently used
    /// for debug output.
    ///
    /// This return type is not stable, and will likely be changed when a more principled way to access
    /// relevant statistics is implemented
    #[cfg_attr(docsrs, doc(hidden))]
    #[deprecated(
        note = "render_to_texture should be preferred, as the _async version has no stability guarantees"
    )]
    pub async fn render_to_texture_async(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        texture: &TextureView,
        params: &RenderParams,
        debug_layers: DebugLayers,
    ) -> Result<Option<BumpAllocators>> {
        if cfg!(not(feature = "debug_layers")) && !debug_layers.is_empty() {
            static HAS_WARNED: AtomicBool = AtomicBool::new(false);
            if !HAS_WARNED.swap(true, std::sync::atomic::Ordering::Release) {
                log::warn!(
                    "Requested debug layers {debug_layers:?} but `debug_layers` feature is not enabled"
                );
            }
        }

        let result = self
            .render_to_texture_async_internal(device, queue, scene, texture, params)
            .await?;

        #[cfg(feature = "debug_layers")]
        {
            let mut recording = Recording::default();
            let target_proxy = recording::ImageProxy::new(
                params.width,
                params.height,
                recording::ImageFormat::Rgba8,
            );
            if let Some(captured) = result.captured {
                let bump = result.bump.as_ref().unwrap();
                // TODO: We could avoid this download if `DebugLayers::VALIDATION` is unset.
                let downloads = DebugDownloads::map(&self.engine, &captured, bump).await?;
                self.debug.render(
                    &mut recording,
                    target_proxy,
                    &captured,
                    bump,
                    params,
                    &downloads,
                    debug_layers,
                );

                // TODO: this sucks. better to release everything in a helper
                // TODO: it would be much better to have a way to safely destroy a buffer.
                self.engine.free_download(captured.lines);
                captured.release_buffers(&mut recording);
            }
            let external_resources = [ExternalResource::Image(target_proxy, texture)];
            self.engine.run_recording(
                device,
                queue,
                &recording,
                &external_resources,
                "render_to_texture_async debug layers",
                #[cfg(feature = "wgpu-profiler")]
                &mut self.profiler,
                None,
            )?;
        }

        #[cfg(feature = "wgpu-profiler")]
        {
            self.profiler.end_frame().unwrap();
            if let Some(result) = self
                .profiler
                .process_finished_frame(queue.get_timestamp_period())
            {
                self.profile_result = Some(result);
            }
        }

        Ok(result.bump)
    }

    async fn render_to_texture_async_internal(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        texture: &TextureView,
        params: &RenderParams,
    ) -> Result<RenderResult> {
        let mut render = Render::with_vibrant_identity(self.vibrant_identity.0);
        let encoding = scene.encoding();
        // TODO: turn this on; the download feature interacts with CPU dispatch.
        // Currently this is always enabled when the `debug_layers` setting is enabled as the bump
        // counts are used for debug visualiation.
        let robust = cfg!(feature = "debug_layers");
        let recording = render.render_encoding_coarse(
            encoding,
            &mut self.resolver,
            &self.shaders,
            &mut self.image_atlas,
            params,
            robust,
        );
        let target = render.out_image();
        let bump_buf = render.bump_buf();
        #[cfg(feature = "debug_layers")]
        let captured = render.take_captured_buffers();
        self.engine.run_recording(
            device,
            queue,
            &recording,
            &[],
            "t_async_coarse",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
            None,
        )?;

        let mut bump: Option<BumpAllocators> = None;
        if let Some(bump_buf) = self.engine.get_download(bump_buf) {
            let buf_slice = bump_buf.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            receiver.receive().await.expect("channel was closed")?;
            let mapped = buf_slice.get_mapped_range().unwrap();
            bump = Some(bytemuck::pod_read_unaligned(&mapped));
        }
        // TODO: apply logic to determine whether we need to rerun coarse, and also
        // allocate the blend stack as needed.
        self.engine.free_download(bump_buf);
        // Maybe clear to reuse allocation?
        let mut recording = Recording::default();
        render.record_fine(&self.shaders, &mut recording);
        let external_resources = [
            ExternalResource::Image(target, texture),
            ExternalResource::Buffer(self.vibrant_identity.0, &self.vibrant_identity.1),
        ];
        self.engine.run_recording(
            device,
            queue,
            &recording,
            &external_resources,
            "t_async_fine",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
            None,
        )?;
        Ok(RenderResult {
            bump,
            #[cfg(feature = "debug_layers")]
            captured,
        })
    }
}
#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
pub(crate) struct DebugDownloads<'a> {
    pub lines: wgpu::BufferSlice<'a>,
}

#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
impl<'a> DebugDownloads<'a> {
    pub async fn map(
        engine: &'a WgpuEngine,
        captured: &render::CapturedBuffers,
        bump: &BumpAllocators,
    ) -> Result<DebugDownloads<'a>> {
        use vello_encoding::LineSoup;

        let Some(lines_buf) = engine.get_download(captured.lines) else {
            return Err(Error::DownloadError("linesoup"));
        };

        let lines = lines_buf.slice(..bump.lines as u64 * size_of::<LineSoup>() as u64);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        lines.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        receiver.receive().await.expect("channel was closed")?;
        Ok(Self { lines })
    }
}
