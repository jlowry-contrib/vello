// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Times the stock render path on a fixed scene and prints one JSON line.
//!
//! Only `Renderer::new` and `render_to_texture` are used, so the same source builds on any
//! revision and their medians compare the revisions' stock frames.

use std::num::NonZeroUsize;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use vello::kurbo::{Affine, Rect};
use vello::peniko::{Color, Fill, Gradient};
use vello::wgpu;
use vello::{AaConfig, AaSupport, RenderParams, Renderer, RendererOptions, Scene};

/// The target: an iPhone 17 Pro's screen in device pixels, 402×874 pt at 3×.
const WIDTH: u32 = 1206;
/// See [`WIDTH`].
const HEIGHT: u32 = 2622;

/// Frames rendered before timing starts, so pipelines and allocations settle.
const WARM_UP: usize = 40;

/// Frames rendered in all, warm-up included.
const FRAMES: usize = 440;

/// The adapter the timings are taken on.
const ADAPTER: &str = "Radeon Pro Vega 20";

fn main() -> Result<()> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::METAL));
    let adapter = adapters
        .into_iter()
        .find(|adapter| adapter.get_info().name.contains(ADAPTER))
        .with_context(|| format!("no Metal adapter named {ADAPTER:?}"))?;
    let features =
        adapter.features() & (wgpu::Features::CLEAR_TEXTURE | wgpu::Features::PIPELINE_CACHE);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("stock_frame_time"),
        required_features: features,
        ..Default::default()
    }))
    .context("could not open the device")?;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stock_frame_time.target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut renderer = Renderer::new(
        &device,
        RendererOptions {
            use_cpu: false,
            antialiasing_support: AaSupport {
                area: false,
                msaa8: false,
                msaa16: true,
            },
            num_init_threads: NonZeroUsize::new(1),
            pipeline_cache: None,
        },
    )
    .map_err(|error| anyhow!("could not create the renderer: {error}"))?;
    let params = RenderParams {
        base_color: Color::WHITE,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Msaa16,
    };
    let scene = bench_scene();
    let mut cpu = Vec::with_capacity(FRAMES - WARM_UP);
    let mut wait = Vec::with_capacity(FRAMES - WARM_UP);
    for frame in 0..FRAMES {
        let start = Instant::now();
        renderer
            .render_to_texture(&device, &queue, &scene, &view, &params)
            .map_err(|error| anyhow!("render failed: {error}"))?;
        let submitted = Instant::now();
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("waiting for the GPU failed")?;
        if frame >= WARM_UP {
            cpu.push((submitted - start).as_secs_f64() * 1000.0);
            wait.push(submitted.elapsed().as_secs_f64() * 1000.0);
        }
    }
    println!(
        "{{ \"rev\": {:?}, \"frames\": {}, \"wait_ms_median\": {}, \"cpu_ms_median\": {} }}",
        revision(),
        wait.len(),
        median(&mut wait),
        median(&mut cpu)
    );
    Ok(())
}

/// Builds the split bench's stock scene: 2000 rectangles and 20 gradients, the same on every run.
fn bench_scene() -> Scene {
    let mut scene = Scene::new();
    let mut rng = Lcg(1);
    for _ in 0..2000 {
        let colour = rng.colour();
        let rect = rng.rect(160.0);
        scene.fill(Fill::NonZero, Affine::IDENTITY, colour, None, &rect);
    }
    for _ in 0..20 {
        let rect = rng.rect(600.0);
        let gradient = Gradient::new_linear((rect.x0, rect.y0), (rect.x1, rect.y1))
            .with_stops([rng.colour(), rng.colour(), rng.colour()].as_slice());
        scene.fill(Fill::NonZero, Affine::IDENTITY, &gradient, None, &rect);
    }
    scene
}

/// A generator whose sequence is the same on every device, as in the split bench.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        // Knuth's MMIX multiplier and increment.
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn unit(&mut self) -> f64 {
        // The top 53 bits fill an f64 mantissa exactly.
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Returns a colour, translucent one time in four so blending is exercised too.
    fn colour(&mut self) -> Color {
        let [r, g, b, a, translucent, ..] = self.next_u64().to_be_bytes();
        let alpha = if translucent < 64 { a.max(32) } else { 255 };
        Color::from_rgba8(r, g, b, alpha)
    }

    fn rect(&mut self, max_side: f64) -> Rect {
        let x = self.unit() * f64::from(WIDTH);
        let y = self.unit() * f64::from(HEIGHT);
        let w = 4.0 + self.unit() * max_side;
        let h = 4.0 + self.unit() * max_side;
        Rect::new(x, y, x + w, y + h)
    }
}

/// Returns the median of `samples`, averaging the middle two of an even count.
fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    }
}

/// Returns `git describe` of the checkout this example was built from, or `unknown`.
fn revision() -> String {
    Command::new("git")
        .args([
            "-C",
            env!("CARGO_MANIFEST_DIR"),
            "describe",
            "--always",
            "--dirty",
        ])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map_or_else(|| "unknown".to_owned(), |rev| rev.trim().to_owned())
}
