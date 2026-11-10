// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renders from the upstream code, saved so later changes can prove the stock path unchanged.
//!
//! Run with `GLASS_BASELINE_WRITE=1` on the unpatched code to write
//! `target/glass-baseline/<name>-<aa>.png` (relative to this crate's directory).

use std::path::{Path, PathBuf};

use vello::kurbo::{Affine, Circle, Rect};
use vello::peniko::{Color, Fill, Mix, color::palette};
use vello::{AaConfig, Scene};
use vello_research_tests::glass::background;
use vello_research_tests::{TestParams, get_scene_image, write_png_to_file};

const SIZE: u32 = 512;

fn layer(scene: &mut Scene, alpha: f32, clip: Rect) {
    scene.push_layer(Fill::NonZero, Mix::Normal, alpha, Affine::IDENTITY, &clip);
}

fn baseline_scenes() -> [(&'static str, Scene); 4] {
    let mut plain = Scene::new();
    background(&mut plain);

    let mut half = Scene::new();
    background(&mut half);
    layer(&mut half, 0.5, Rect::new(64.0, 64.0, 448.0, 448.0));
    half.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgba8(20, 60, 220, 255),
        None,
        &Circle::new((256.0, 256.0), 220.0),
    );
    half.pop_layer();

    let mut nested = Scene::new();
    background(&mut nested);
    for i in 0..5_u8 {
        let inset = 24.0 + 32.0 * f64::from(i);
        layer(
            &mut nested,
            0.9,
            Rect::new(inset, inset, 512.0 - inset, 512.0 - inset),
        );
        nested.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::from_rgba8(40 * i, 90, 200 - 30 * i, 160),
            None,
            &Rect::new(inset, inset, 256.0, 512.0 - inset),
        );
    }
    for _ in 0..5 {
        nested.pop_layer();
    }

    let mut circle = Scene::new();
    background(&mut circle);
    circle.push_clip_layer(
        Fill::NonZero,
        Affine::IDENTITY,
        &Circle::new((256.0, 256.0), 180.0),
    );
    circle.fill(
        Fill::NonZero,
        Affine::translate((40.0, -30.0)),
        Color::from_rgba8(250, 200, 40, 255),
        None,
        &Rect::new(0.0, 0.0, 512.0, 512.0),
    );
    circle.pop_layer();

    [
        ("background", plain),
        ("half-layer", half),
        ("nested-layers", nested),
        ("circle-clip", circle),
    ]
}

fn baseline_path(name: &str, aa: &str) -> PathBuf {
    Path::new("target/glass-baseline").join(format!("{name}-{aa}.png"))
}

fn read_png(path: &Path) -> Vec<u8> {
    image::open(path)
        .unwrap_or_else(|e| {
            panic!(
                "missing baseline {}: regenerate it from f3000c8 with GLASS_BASELINE_WRITE=1 ({e})",
                path.display()
            )
        })
        .to_rgba8()
        .into_raw()
}

#[test]
fn stock_path_matches_the_upstream_baseline() {
    let write = std::env::var_os("GLASS_BASELINE_WRITE").is_some_and(|v| v == "1");
    if write {
        std::fs::create_dir_all("target/glass-baseline").unwrap();
    }
    for (name, scene) in baseline_scenes() {
        for (aa, label) in [(AaConfig::Area, "area"), (AaConfig::Msaa16, "msaa16")] {
            let params = TestParams {
                base_color: Some(palette::css::WHITE),
                anti_aliasing: aa,
                ..TestParams::new(format!("glass_baseline_{name}_{label}"), SIZE, SIZE)
            };
            let image = pollster::block_on(get_scene_image(&params, &scene)).unwrap();
            let path = baseline_path(name, label);
            if write {
                write_png_to_file(&params, &path, &image, None, false).unwrap();
            } else {
                let expected = read_png(&path);
                assert!(
                    image.data.data() == expected.as_slice(),
                    "{name} with {label} differs from {}",
                    path.display()
                );
            }
        }
    }
}
