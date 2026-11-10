// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Deterministic scenes and helpers for the glass tests.

use std::num::NonZeroUsize;

use vello::glass::{GlassFrame, GlassPasses, GlassPhase, GlassResumeFormat, MarkerPlacement};
use vello::kurbo::{Affine, Rect};
use vello::peniko::{Color, Fill, Gradient, color::palette};
use vello::util::block_on_wgpu;
use vello::wgpu::{self, CommandEncoder, Device, Queue, Texture};
use vello::{AaConfig, RenderParams, Renderer, RendererOptions, Scene};

use crate::WGPU_DEVICE_QUEUE;

/// Width and height of the area `background` covers, in pixels.
pub const BACKGROUND_SIZE: f64 = 512.0;

/// A linear congruential generator, so scenes are identical on every run.
#[derive(Clone, Debug)]
pub struct Lcg(u64);

impl Lcg {
    /// Creates a generator from `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Returns the next state.
    pub fn next_u64(&mut self) -> u64 {
        // Knuth's MMIX multiplier and increment.
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Returns the next value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // The top 53 bits fill an f64 mantissa exactly.
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Returns an opaque colour from the high bytes of the next state.
    pub fn color(&mut self) -> Color {
        let [r, g, b, ..] = self.next_u64().to_be_bytes();
        Color::from_rgb8(r, g, b)
    }
}

/// Draws a linear gradient and 40 rectangles from an LCG with seed 1.
pub fn background(scene: &mut Scene) {
    let mut rng = Lcg::new(1);
    let full = Rect::new(0.0, 0.0, BACKGROUND_SIZE, BACKGROUND_SIZE);
    let gradient = Gradient::new_linear((0.0, 0.0), (BACKGROUND_SIZE, BACKGROUND_SIZE))
        .with_stops([rng.color(), rng.color(), rng.color()].as_slice());
    scene.fill(Fill::NonZero, Affine::IDENTITY, &gradient, None, &full);
    for _ in 0..40 {
        let x = rng.next_f64() * BACKGROUND_SIZE;
        let y = rng.next_f64() * BACKGROUND_SIZE;
        let w = 8.0 + rng.next_f64() * 120.0;
        let h = 8.0 + rng.next_f64() * 120.0;
        let color = rng.color();
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            color,
            None,
            &Rect::new(x, y, x + w, y + h),
        );
    }
}

/// An atlas read back from the GPU: `width`×`height` RGBA8 texels.
#[derive(Clone, Debug)]
pub struct Atlas {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Atlas {
    /// Returns the texel at `(x, y)`.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        assert!(
            x < self.width && y < self.height,
            "({x}, {y}) lies outside the {}x{} atlas",
            self.width,
            self.height
        );
        let i = ((y * self.width + x) * 4) as usize;
        self.data[i..i + 4].try_into().unwrap()
    }
}

/// Glass passes that copy the backdrop atlas out and record what they were given.
#[derive(Debug, Default)]
pub struct CaptureBackdrop {
    /// The marker placements `prepare` saw.
    pub frame: Vec<MarkerPlacement>,
    /// The phase count `prepare` saw.
    pub frame_phases: u32,
    /// The phase index of each `record` call, in order.
    pub phases: Vec<u32>,
    /// The markers of every `record` call, in order.
    pub placements: Vec<MarkerPlacement>,
    capture: Option<(wgpu::Buffer, u32, u32)>,
}

impl CaptureBackdrop {
    /// Reads back the backdrop atlas as the last `record` call saw it.
    pub fn backdrop(&self) -> Atlas {
        let (buffer, width, height) = self.capture.as_ref().expect("no phase was recorded");
        let (device, _) = &*WGPU_DEVICE_QUEUE;
        let data = read_buffer(device, buffer, *width, *height);
        Atlas {
            width: *width,
            height: *height,
            data,
        }
    }
}

impl GlassPasses for CaptureBackdrop {
    fn prepare(&mut self, _device: &Device, _queue: &Queue, frame: &GlassFrame<'_>) {
        self.frame = frame.markers.to_vec();
        self.frame_phases = frame.phases;
    }

    fn record(&mut self, device: &Device, encoder: &mut CommandEncoder, phase: &GlassPhase<'_>) {
        self.phases.push(phase.index);
        self.placements.extend_from_slice(phase.markers);
        let buffer = copy_to_buffer(device, encoder, phase.backdrop_atlas);
        let atlas = phase.backdrop_atlas;
        self.capture = Some((buffer, atlas.width(), atlas.height()));
    }
}

/// Glass passes that do nothing, leaving the glass atlas as it was.
#[derive(Debug, Default)]
pub struct NoGlass;

impl GlassPasses for NoGlass {
    fn prepare(&mut self, _device: &Device, _queue: &Queue, _frame: &GlassFrame<'_>) {}

    fn record(&mut self, _device: &Device, _encoder: &mut CommandEncoder, _phase: &GlassPhase<'_>) {
    }
}

/// Returns the largest per-byte difference between `a` and `b`.
pub fn max_delta(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "compared buffers differ in length");
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

/// Renders `scene` on opaque white with the stock path and reads it back as RGBA8.
pub fn render_scene(width: u32, height: u32, aa: AaConfig, scene: &Scene) -> Vec<u8> {
    let (device, queue) = &*WGPU_DEVICE_QUEUE;
    let mut renderer = renderer(aa, false);
    let target = target(device, width, height);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    renderer
        .render_to_texture(device, queue, scene, &view, &params(width, height, aa))
        .unwrap();
    read_texture(device, queue, &target)
}

/// Renders `build`'s scene unsplit without markers, then split with markers.
///
/// `build` receives whether to push the glass markers, and `configure` sets up the renderer
/// first. Both renders are on opaque white, read back as RGBA8.
pub fn render_pair(
    width: u32,
    height: u32,
    aa: AaConfig,
    resume: GlassResumeFormat,
    configure: impl Fn(&mut Renderer),
    build: impl Fn(&mut Scene, bool),
    passes: &mut dyn GlassPasses,
) -> (Vec<u8>, Vec<u8>) {
    let mut renderer = renderer(aa, false);
    renderer.set_glass_resume_format(resume);
    configure(&mut renderer);
    pair(&mut renderer, width, height, aa, build, passes)
}

/// Renders as [`render_pair`] does, with the CPU stages and the default settings.
pub fn render_pair_cpu(
    width: u32,
    height: u32,
    build: impl Fn(&mut Scene, bool),
    passes: &mut dyn GlassPasses,
) -> (Vec<u8>, Vec<u8>) {
    let mut renderer = renderer(AaConfig::Area, true);
    pair(&mut renderer, width, height, AaConfig::Area, build, passes)
}

fn pair(
    renderer: &mut Renderer,
    width: u32,
    height: u32,
    aa: AaConfig,
    build: impl Fn(&mut Scene, bool),
    passes: &mut dyn GlassPasses,
) -> (Vec<u8>, Vec<u8>) {
    let (device, queue) = &*WGPU_DEVICE_QUEUE;
    let params = params(width, height, aa);
    let [unsplit, split] = [false, true].map(|marked| {
        let mut scene = Scene::new();
        build(&mut scene, marked);
        let target = target(device, width, height);
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        if marked {
            renderer
                .render_to_texture_with_glass(device, queue, &scene, &view, &params, &mut *passes)
                .unwrap();
        } else {
            renderer
                .render_to_texture(device, queue, &scene, &view, &params)
                .unwrap();
        }
        read_texture(device, queue, &target)
    });
    (unsplit, split)
}

fn renderer(aa: AaConfig, use_cpu: bool) -> Renderer {
    let (device, _) = &*WGPU_DEVICE_QUEUE;
    Renderer::new(
        device,
        RendererOptions {
            use_cpu,
            num_init_threads: NonZeroUsize::new(1),
            antialiasing_support: std::iter::once(aa).collect(),
            pipeline_cache: None,
        },
    )
    .unwrap()
}

fn params(width: u32, height: u32, aa: AaConfig) -> RenderParams {
    RenderParams {
        base_color: palette::css::WHITE,
        width,
        height,
        antialiasing_method: aa,
    }
}

fn target(device: &Device, width: u32, height: u32) -> Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("glass test target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Bytes per row of a readback, padded to wgpu's copy alignment.
fn padded_row(width: u32) -> u32 {
    (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

fn copy_to_buffer(
    device: &Device,
    encoder: &mut CommandEncoder,
    texture: &Texture,
) -> wgpu::Buffer {
    let row = padded_row(texture.width());
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("glass test readback"),
        size: u64::from(row) * u64::from(texture.height()),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    buffer
}

fn read_texture(device: &Device, queue: &Queue, texture: &Texture) -> Vec<u8> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("glass test readback"),
    });
    let buffer = copy_to_buffer(device, &mut encoder, texture);
    queue.submit([encoder.finish()]);
    read_buffer(device, &buffer, texture.width(), texture.height())
}

fn read_buffer(device: &Device, buffer: &wgpu::Buffer, width: u32, height: u32) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    block_on_wgpu(device, receiver.receive())
        .expect("channel was closed")
        .unwrap();
    let row = padded_row(width) as usize;
    let data = slice.get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height as usize {
        out.extend_from_slice(&data[y * row..y * row + width as usize * 4]);
    }
    drop(data);
    buffer.unmap();
    out
}
