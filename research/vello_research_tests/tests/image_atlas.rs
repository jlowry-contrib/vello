// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The persistent image atlas keeps the images the resolver believes resident.

use std::sync::Arc;

use vello::kurbo::{Affine, Rect};
use vello::peniko::{Blob, Color, Fill, ImageAlphaType, ImageBrush, ImageData, ImageFormat};
use vello::{AaConfig, AaSupport, Renderer, Scene};
use vello_research_tests::glass::{gpu_renderer, render_frame};

const SIZE: u32 = 16;
const RED: [u8; 4] = [255, 0, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];

fn red_image() -> ImageData {
    let pixels = RED.repeat((SIZE * SIZE) as usize);
    ImageData {
        data: Blob::new(Arc::new(pixels)),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width: SIZE,
        height: SIZE,
    }
}

fn image_scene(image: &ImageData) -> Scene {
    let mut scene = Scene::new();
    scene.draw_image(&ImageBrush::new(image.clone()), Affine::IDENTITY);
    scene
}

/// A scene with no gradients, glyphs or images, which resolves without patches.
fn solid_scene() -> Scene {
    let mut scene = Scene::new();
    let rect = Rect::new(0.0, 0.0, f64::from(SIZE), f64::from(SIZE));
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::new([0.0, 0.0, 1.0, 1.0]),
        None,
        &rect,
    );
    scene
}

fn centre(pixels: &[u8]) -> [u8; 4] {
    let at = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
    pixels[at..at + 4].try_into().unwrap()
}

fn render(renderer: &mut Renderer, scene: &Scene) -> [u8; 4] {
    centre(&render_frame(
        renderer,
        SIZE,
        SIZE,
        AaConfig::Area,
        scene,
        None,
    ))
}

/// An image drawn again after a frame without patches still draws.
///
/// A frame without patches reports no atlas, so the renderer replaces the atlas texture
/// while the image cache still holds the image as resident.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn image_draws_after_a_frame_without_patches() {
    let mut renderer = gpu_renderer(AaSupport::area_only());
    let image = red_image();

    assert_eq!(render(&mut renderer, &image_scene(&image)), RED);
    assert_eq!(render(&mut renderer, &solid_scene()), BLUE);
    assert_eq!(render(&mut renderer, &image_scene(&image)), RED);
}
