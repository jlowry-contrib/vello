// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Load rendering shaders.

#[cfg(feature = "wgpu")]
use std::borrow::Cow;

#[cfg(feature = "wgpu")]
use wgpu::Device;

use crate::ShaderId;

#[cfg(feature = "wgpu")]
use crate::{
    AaSupport, Error, RendererOptions,
    recording::{BindType, ImageFormat},
    wgpu_engine::WgpuEngine,
};

/// Glass fine permutations and the dispatch setups, built on the first glass frame.
///
/// A fine variant is `None` when its AA mode isn't in the renderer's `AaSupport`.
pub(crate) struct GlassShaders {
    pub(crate) fine_area_lite: Option<ShaderId>,
    pub(crate) fine_msaa8_lite: Option<ShaderId>,
    pub(crate) fine_msaa16_lite: Option<ShaderId>,
    pub(crate) fine_area_start: Option<ShaderId>,
    pub(crate) fine_msaa8_start: Option<ShaderId>,
    pub(crate) fine_msaa16_start: Option<ShaderId>,
    pub(crate) fine_area_resume: Option<ShaderId>,
    pub(crate) fine_msaa8_resume: Option<ShaderId>,
    pub(crate) fine_msaa16_resume: Option<ShaderId>,
    pub(crate) glass_resume_setup: ShaderId,
    pub(crate) glass_start_setup: ShaderId,
    /// Whether the lite permutations count their commands, for tests.
    #[cfg_attr(
        not(feature = "wgpu"),
        expect(
            dead_code,
            reason = "only the wgpu renderer rebuilds the glass shaders"
        )
    )]
    pub(crate) lite_counted: bool,
}

/// Returns the shader table: compiled at build time, or from source with `hot_reload`.
#[cfg(all(feature = "wgpu", feature = "hot_reload"))]
fn shader_table()
-> Result<std::collections::HashMap<String, vello_shaders::compile::ShaderInfo>, Error> {
    Ok(vello_shaders::compile::ShaderInfo::from_default()?)
}

/// Returns the shader table: compiled at build time, or from source with `hot_reload`.
#[cfg(all(feature = "wgpu", not(feature = "hot_reload")))]
fn shader_table() -> Result<vello_shaders::Shaders<'static>, Error> {
    Ok(vello_shaders::SHADERS)
}

/// Takes the WGSL source of shader `$name` out of a [`shader_table`].
#[cfg(feature = "wgpu")]
macro_rules! shader_source {
    ($shaders:ident, $name:ident) => {{
        #[cfg(feature = "hot_reload")]
        let source = $shaders
            .remove(stringify!($name))
            .expect(stringify!($name))
            .source
            .into();
        #[cfg(not(feature = "hot_reload"))]
        let source = $shaders.$name.wgsl.code;
        source
    }};
}

/// The bindings of the stock fine shaders, ending with the mask LUT that only MSAA uses.
#[cfg(feature = "wgpu")]
const FINE_RESOURCES: [BindType; 9] = [
    BindType::Uniform,
    BindType::BufReadOnly,
    BindType::BufReadOnly,
    BindType::BufReadOnly,
    BindType::Buffer,
    BindType::Image(ImageFormat::Rgba8),
    BindType::ImageRead(ImageFormat::Rgba8),
    BindType::ImageRead(ImageFormat::Rgba8),
    // Mask LUT buffer, used only when MSAA is enabled.
    BindType::BufReadOnly,
];

/// Builds the glass fine permutations for the AA modes in `aa_support`, and the dispatch setups.
///
/// With `lite_counted`, the lite permutations count their commands in `glass_state`, for tests.
#[cfg(feature = "wgpu")]
pub(crate) fn glass_shaders(
    device: &Device,
    engine: &mut WgpuEngine,
    aa_support: AaSupport,
    lite_counted: bool,
) -> Result<GlassShaders, Error> {
    use crate::wgpu_engine::CpuShaderType;
    use BindType::*;

    #[cfg_attr(
        not(feature = "hot_reload"),
        expect(unused_mut, reason = "only the hot-reload table is consumed")
    )]
    let mut shaders = shader_table()?;
    // Glass needs the GPU stages, so no glass shader has a CPU mirror.
    let mut add = |label: &'static str, source: Cow<'static, str>, bindings: &[BindType]| {
        engine.add_compute_shader(device, label, source, bindings, CpuShaderType::Missing)
    };

    // Marker table, glass state, backdrop atlas and glass atlas, after the mask LUT if any.
    let glass_resources = [
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
    ];
    let area_resources = &FINE_RESOURCES[..FINE_RESOURCES.len() - 1];
    let area = [area_resources, &glass_resources].concat();
    let msaa = [&FINE_RESOURCES[..], &glass_resources].concat();

    // The lite permutations bind stock fine's resources plus the glass state.
    let area_lite = [area_resources, &[Buffer]].concat();
    let msaa_lite = [&FINE_RESOURCES[..], &[Buffer]].concat();

    let [fine_area_lite, fine_area_start, fine_area_resume] = if aa_support.area {
        [
            add(
                "vello.fine_area_lite",
                if lite_counted {
                    shader_source!(shaders, fine_area_lite_counted)
                } else {
                    shader_source!(shaders, fine_area_lite)
                },
                &area_lite,
            ),
            add(
                "vello.fine_area_start",
                shader_source!(shaders, fine_area_start),
                &area,
            ),
            add(
                "vello.fine_area_resume",
                shader_source!(shaders, fine_area_resume),
                &area,
            ),
        ]
        .map(Some)
    } else {
        [None; 3]
    };
    let [fine_msaa8_lite, fine_msaa8_start, fine_msaa8_resume] = if aa_support.msaa8 {
        [
            add(
                "vello.fine_msaa8_lite",
                if lite_counted {
                    shader_source!(shaders, fine_msaa8_lite_counted)
                } else {
                    shader_source!(shaders, fine_msaa8_lite)
                },
                &msaa_lite,
            ),
            add(
                "vello.fine_msaa8_start",
                shader_source!(shaders, fine_msaa8_start),
                &msaa,
            ),
            add(
                "vello.fine_msaa8_resume",
                shader_source!(shaders, fine_msaa8_resume),
                &msaa,
            ),
        ]
        .map(Some)
    } else {
        [None; 3]
    };
    let [fine_msaa16_lite, fine_msaa16_start, fine_msaa16_resume] = if aa_support.msaa16 {
        [
            add(
                "vello.fine_msaa16_lite",
                if lite_counted {
                    shader_source!(shaders, fine_msaa16_lite_counted)
                } else {
                    shader_source!(shaders, fine_msaa16_lite)
                },
                &msaa_lite,
            ),
            add(
                "vello.fine_msaa16_start",
                shader_source!(shaders, fine_msaa16_start),
                &msaa,
            ),
            add(
                "vello.fine_msaa16_resume",
                shader_source!(shaders, fine_msaa16_resume),
                &msaa,
            ),
        ]
        .map(Some)
    } else {
        [None; 3]
    };
    let glass_resume_setup = add(
        "vello.glass_resume_setup",
        shader_source!(shaders, glass_resume_setup),
        &[Uniform, Buffer, Buffer],
    );
    let glass_start_setup = add(
        "vello.glass_start_setup",
        shader_source!(shaders, glass_start_setup),
        &[Uniform, Buffer, Buffer],
    );

    Ok(GlassShaders {
        fine_area_lite,
        fine_msaa8_lite,
        fine_msaa16_lite,
        fine_area_start,
        fine_msaa8_start,
        fine_msaa16_start,
        fine_area_resume,
        fine_msaa8_resume,
        fine_msaa16_resume,
        glass_resume_setup,
        glass_start_setup,
        lite_counted,
    })
}

// Shaders for the full pipeline
pub struct FullShaders {
    pub pathtag_reduce: ShaderId,
    pub pathtag_reduce2: ShaderId,
    pub pathtag_scan1: ShaderId,
    pub pathtag_scan: ShaderId,
    pub pathtag_scan_large: ShaderId,
    pub bbox_clear: ShaderId,
    pub flatten: ShaderId,
    pub draw_reduce: ShaderId,
    pub draw_leaf: ShaderId,
    pub clip_reduce: ShaderId,
    pub clip_leaf: ShaderId,
    pub binning: ShaderId,
    pub tile_alloc: ShaderId,
    pub backdrop: ShaderId,
    pub path_count_setup: ShaderId,
    pub path_count: ShaderId,
    pub coarse: ShaderId,
    pub path_tiling_setup: ShaderId,
    pub path_tiling: ShaderId,
    pub fine_area: Option<ShaderId>,
    pub fine_msaa8: Option<ShaderId>,
    pub fine_msaa16: Option<ShaderId>,
    // 2-level dispatch works for CPU pathtag scan even for large
    // inputs, 3-level is not yet implemented.
    pub pathtag_is_cpu: bool,
}

#[cfg(feature = "wgpu")]
pub(crate) fn full_shaders(
    device: &Device,
    engine: &mut WgpuEngine,
    options: &RendererOptions,
) -> Result<FullShaders, Error> {
    use crate::wgpu_engine::CpuShaderType;
    use BindType::*;

    let mut force_gpu = false;
    let force_gpu_from: Option<&str> = None;
    // Uncomment this to force use of GPU shaders from the specified shader and later even
    // if `engine.use_cpu` is specified.
    //let force_gpu_from = Some("binning");

    #[cfg_attr(
        not(feature = "hot_reload"),
        expect(unused_mut, reason = "only the hot-reload table is consumed")
    )]
    let mut shaders = shader_table()?;

    macro_rules! add_shader {
        ($name:ident, $label:expr, $bindings:expr, $cpu:expr) => {{
            if force_gpu_from == Some(stringify!($name)) {
                force_gpu = true;
            }
            let source = shader_source!(shaders, $name);
            engine.add_compute_shader(
                device,
                concat!("vello.", $label),
                source,
                &$bindings,
                if force_gpu {
                    CpuShaderType::Missing
                } else {
                    $cpu
                },
            )
        }};
        ($name:ident, $bindings:expr, $cpu:expr) => {{ add_shader!($name, stringify!($name), $bindings, $cpu) }};
        ($name:ident, $bindings:expr) => {
            add_shader!(
                $name,
                $bindings,
                CpuShaderType::Present(vello_shaders::cpu::$name)
            )
        };
    }

    let pathtag_reduce = add_shader!(pathtag_reduce, [Uniform, BufReadOnly, Buffer]);
    let pathtag_reduce2 = add_shader!(
        pathtag_reduce2,
        [BufReadOnly, Buffer],
        CpuShaderType::Skipped
    );
    let pathtag_scan1 = add_shader!(
        pathtag_scan1,
        [BufReadOnly, BufReadOnly, Buffer],
        CpuShaderType::Skipped
    );
    let pathtag_scan = add_shader!(
        pathtag_scan_small,
        [Uniform, BufReadOnly, BufReadOnly, Buffer],
        CpuShaderType::Present(vello_shaders::cpu::pathtag_scan)
    );
    let pathtag_scan_large = add_shader!(
        pathtag_scan_large,
        [Uniform, BufReadOnly, BufReadOnly, Buffer],
        CpuShaderType::Skipped
    );
    let bbox_clear = add_shader!(bbox_clear, [Uniform, Buffer]);
    let flatten = add_shader!(
        flatten,
        [Uniform, BufReadOnly, BufReadOnly, Buffer, Buffer, Buffer]
    );
    let draw_reduce = add_shader!(draw_reduce, [Uniform, BufReadOnly, Buffer]);
    let draw_leaf = add_shader!(
        draw_leaf,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
        ]
    );
    let clip_reduce = add_shader!(clip_reduce, [BufReadOnly, BufReadOnly, Buffer, Buffer]);
    let clip_leaf = add_shader!(
        clip_leaf,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
        ]
    );
    let binning = add_shader!(
        binning,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
            Buffer,
        ]
    );
    let tile_alloc = add_shader!(
        tile_alloc,
        [Uniform, BufReadOnly, BufReadOnly, Buffer, Buffer, Buffer]
    );
    let path_count_setup = add_shader!(path_count_setup, [Buffer, Buffer]);
    let path_count = add_shader!(
        path_count,
        [Uniform, Buffer, BufReadOnly, BufReadOnly, Buffer, Buffer]
    );
    let backdrop = add_shader!(
        backdrop_dyn,
        [Uniform, Buffer, BufReadOnly, Buffer],
        CpuShaderType::Present(vello_shaders::cpu::backdrop)
    );
    let coarse = add_shader!(
        coarse,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
        ]
    );
    let path_tiling_setup = add_shader!(path_tiling_setup, [Buffer, Buffer, Buffer]);
    let path_tiling = add_shader!(
        path_tiling,
        [
            Buffer,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
        ]
    );
    let fine_resources = FINE_RESOURCES;
    let area_resources = &fine_resources[..fine_resources.len() - 1];

    let aa_support = &options.antialiasing_support;
    let fine_area = if aa_support.area {
        Some(add_shader!(
            fine_area,
            area_resources,
            CpuShaderType::Missing
        ))
    } else {
        None
    };
    let fine_msaa8 = if aa_support.msaa8 {
        Some(add_shader!(
            fine_msaa8,
            fine_resources,
            CpuShaderType::Missing
        ))
    } else {
        None
    };
    let fine_msaa16 = if aa_support.msaa16 {
        Some(add_shader!(
            fine_msaa16,
            fine_resources,
            CpuShaderType::Missing
        ))
    } else {
        None
    };

    Ok(FullShaders {
        pathtag_reduce,
        pathtag_reduce2,
        pathtag_scan,
        pathtag_scan1,
        pathtag_scan_large,
        bbox_clear,
        flatten,
        draw_reduce,
        draw_leaf,
        clip_reduce,
        clip_leaf,
        binning,
        tile_alloc,
        path_count_setup,
        path_count,
        backdrop,
        coarse,
        path_tiling_setup,
        path_tiling,
        fine_area,
        fine_msaa8,
        fine_msaa16,
        pathtag_is_cpu: options.use_cpu,
    })
}
