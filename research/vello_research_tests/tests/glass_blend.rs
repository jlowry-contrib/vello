// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Plus-darker and vibrant matrix layers.

use std::path::Path;

use vello::glass::{GlassMarker, IRect};
use vello::kurbo::{Affine, Rect};
use vello::peniko::{Color, Fill};
use vello::{AaConfig, AaSupport, Renderer, Scene, VibrantMatrix};
use vello_research_tests::glass::{
    BASELINE_SIZE, NoGlass, baseline_path, baseline_scenes, gpu_renderer, max_delta, render_frame,
};

const SIZE: u32 = 64;
const MODES: [AaConfig; 2] = [AaConfig::Area, AaConfig::Msaa16];
const FULL: Rect = Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64);
/// The layer clip of the vibrant tests, away from every edge.
const CLIP: Rect = Rect::new(16.0, 16.0, 48.0, 48.0);

fn fill(scene: &mut Scene, rgb: [f32; 3], shape: Rect) {
    let [r, g, b] = rgb;
    let color = Color::new([r, g, b, 1.0]);
    scene.fill(Fill::NonZero, Affine::IDENTITY, color, None, &shape);
}

/// Fills each quadrant of the target with its own opaque colour.
fn quadrants(scene: &mut Scene) {
    let half = f64::from(SIZE) / 2.0;
    let colors = [
        [0.9, 0.3, 0.1],
        [0.2, 0.7, 0.4],
        [0.1, 0.2, 0.8],
        [0.6, 0.5, 0.95],
    ];
    for (i, rgb) in colors.into_iter().enumerate() {
        let x = half * (i % 2) as f64;
        let y = half * (i / 2) as f64;
        fill(scene, rgb, Rect::new(x, y, x + half, y + half));
    }
}

fn render(renderer: &mut Renderer, aa: AaConfig, scene: &Scene) -> Vec<u8> {
    render_frame(renderer, SIZE, SIZE, aa, scene, None)
}

/// A made-up matrix that swaps red and blue.
fn red_blue_swap() -> VibrantMatrix {
    VibrantMatrix {
        rows: [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        clamp: (0.0, 1.0),
    }
}

/// A made-up matrix that averages the colour channels.
fn grey() -> VibrantMatrix {
    let third = 1.0 / 3.0;
    VibrantMatrix {
        rows: [
            [third, third, third, 0.0, 0.0],
            [third, third, third, 0.0, 0.0],
            [third, third, third, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        clamp: (0.0, 1.0),
    }
}

/// Pushes a vibrant layer over `clip`, filled opaque so its coverage is full, and pops it.
fn vibrant(scene: &mut Scene, matrix: VibrantMatrix, alpha: f32, clip: Rect) {
    scene.push_vibrant_layer(matrix, alpha, Affine::IDENTITY, &clip);
    fill(scene, [1.0, 1.0, 1.0], clip);
    scene.pop_layer();
}

/// Returns the colour components of every pixel, in `[0, 1]`.
fn pixels(rgba: &[u8]) -> impl Iterator<Item = [f32; 4]> + '_ {
    rgba.chunks(4)
        .map(|p| [p[0], p[1], p[2], p[3]].map(|c| f32::from(c) / 255.0))
}

fn quantize(c: f32) -> f32 {
    (c * 255.0).round() / 255.0
}

/// Asserts every pixel is within 1/255 of `expected(index, pixel of reference)`.
fn assert_close(
    label: &str,
    actual: &[u8],
    reference: &[u8],
    expected: impl Fn(usize, [f32; 4]) -> [f32; 4],
) {
    // Half a step of slack over 1/255 for the rounding of the expected value.
    let tolerance = 1.5 / 255.0;
    for (i, (got, base)) in pixels(actual).zip(pixels(reference)).enumerate() {
        let want = expected(i, base);
        for c in 0..4 {
            assert!(
                (got[c] - want[c]).abs() <= tolerance,
                "{label}: pixel {i} channel {c} is {} for {}",
                got[c],
                want[c]
            );
        }
    }
}

/// The plus-darker law on premultiplied colours, for opaque `cb` and source `cs` at `as_`.
fn plus_darker(cs: [f32; 3], as_: f32, cb: [f32; 3]) -> [f32; 4] {
    let ra = (as_ + 1.0).min(1.0);
    let c = |i: usize| (ra - (as_ * (1.0 - cs[i]) + (1.0 - cb[i]))).max(0.0);
    [c(0), c(1), c(2), ra]
}

fn plus_darker_scene(alpha: f32) -> Scene {
    let mut scene = Scene::new();
    fill(&mut scene, [0.7, 0.2, 1.0], FULL);
    scene.push_layer_plus_darker(alpha, Affine::IDENTITY, &FULL);
    fill(&mut scene, [0.6, 0.6, 0.6], FULL);
    scene.pop_layer();
    scene
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn plus_darker_is_max_zero_src_plus_dst_minus_one() {
    let mut renderer = gpu_renderer(AaSupport::all());
    for aa in MODES {
        let out = render(&mut renderer, aa, &plus_darker_scene(1.0));
        assert_close("opaque", &out, &out, |_, _| [0.3, 0.0, 0.6, 1.0]);
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn plus_darker_with_half_alpha_follows_the_porter_duff_law() {
    let mut renderer = gpu_renderer(AaSupport::all());
    // The colours as the encoding quantizes them.
    let cs = [0.6; 3].map(quantize);
    let cb = [0.7, 0.2, 1.0].map(quantize);
    let want = plus_darker(cs, 0.5, cb);
    for aa in MODES {
        let out = render(&mut renderer, aa, &plus_darker_scene(0.5));
        assert_close("half alpha", &out, &out, |_, _| want);
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn appended_vibrant_layers_keep_their_matrices() {
    let mut renderer = gpu_renderer(AaSupport::all());
    let clips = [
        Rect::new(0.0, 0.0, 32.0, 40.0),
        Rect::new(24.0, 8.0, 64.0, 32.0),
        Rect::new(8.0, 36.0, 56.0, 60.0),
    ];
    // The parent's table is [swap]; the child's is [grey, swap], so appending remaps both.
    let mut appended = Scene::new();
    quadrants(&mut appended);
    vibrant(&mut appended, red_blue_swap(), 1.0, clips[0]);
    let mut child = Scene::new();
    vibrant(&mut child, grey(), 1.0, clips[1]);
    vibrant(&mut child, red_blue_swap(), 0.75, clips[2]);
    appended.append(&child, None);
    assert_eq!(appended.encoding().vibrant.len(), 2);

    let mut direct = Scene::new();
    quadrants(&mut direct);
    vibrant(&mut direct, red_blue_swap(), 1.0, clips[0]);
    vibrant(&mut direct, grey(), 1.0, clips[1]);
    vibrant(&mut direct, red_blue_swap(), 0.75, clips[2]);

    let mut plain = Scene::new();
    quadrants(&mut plain);
    for aa in MODES {
        let appended = render(&mut renderer, aa, &appended);
        let direct = render(&mut renderer, aa, &direct);
        assert_eq!(max_delta(&appended, &direct), 0, "{aa:?}");
        assert_ne!(
            max_delta(&appended, &render(&mut renderer, aa, &plain)),
            0,
            "{aa:?}: the layers changed nothing"
        );
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn vibrant_identity_matrix_leaves_the_backdrop_unchanged() {
    let mut renderer = gpu_renderer(AaSupport::all());
    let mut plain = Scene::new();
    quadrants(&mut plain);
    let mut scene = plain.clone();
    vibrant(&mut scene, VibrantMatrix::IDENTITY, 1.0, CLIP);
    for aa in MODES {
        let reference = render(&mut renderer, aa, &plain);
        let out = render(&mut renderer, aa, &scene);
        assert_eq!(max_delta(&out, &reference), 0, "{aa:?}");
    }
}

/// Returns whether pixel `i` of the target lies inside `CLIP`.
fn in_clip(i: usize) -> bool {
    let (x, y) = ((i % SIZE as usize) as f64, (i / SIZE as usize) as f64);
    CLIP.contains((x + 0.5, y + 0.5))
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn vibrant_matrix_applies_to_the_backdrop_under_the_layer_coverage() {
    let mut renderer = gpu_renderer(AaSupport::all());
    let mut plain = Scene::new();
    quadrants(&mut plain);
    let mut scene = plain.clone();
    vibrant(&mut scene, red_blue_swap(), 0.5, CLIP);
    for aa in MODES {
        let reference = render(&mut renderer, aa, &plain);
        let out = render(&mut renderer, aa, &scene);
        assert_close("swap", &out, &reference, |i, [r, g, b, a]| {
            if in_clip(i) {
                [(r + b) / 2.0, g, (b + r) / 2.0, a]
            } else {
                [r, g, b, a]
            }
        });
        for (i, (got, base)) in out.chunks(4).zip(reference.chunks(4)).enumerate() {
            if !in_clip(i) {
                assert_eq!(got, base, "{aa:?}: pixel {i} outside the clip changed");
            }
        }
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn vibrant_output_is_clamped() {
    let mut renderer = gpu_renderer(AaSupport::all());
    let mut matrix = VibrantMatrix::IDENTITY;
    matrix.rows[0][4] = 2.0;
    // Bounds wider than [0, 1], so the final clamp is the one that applies.
    matrix.clamp = (-4.0, 4.0);
    let mut plain = Scene::new();
    quadrants(&mut plain);
    let mut scene = plain.clone();
    vibrant(&mut scene, matrix, 1.0, CLIP);
    for aa in MODES {
        let reference = render(&mut renderer, aa, &plain);
        let out = render(&mut renderer, aa, &scene);
        assert_close("clamp", &out, &reference, |i, [r, g, b, a]| {
            [if in_clip(i) { 1.0 } else { r }, g, b, a]
        });
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn vibrant_layer_renders_in_a_glass_frame() {
    let mut renderer = gpu_renderer(AaSupport::all());
    let tile = |x0, y0, x1, y1| IRect { x0, y0, x1, y1 };
    let build = |marked: bool| {
        let mut scene = Scene::new();
        quadrants(&mut scene);
        if marked {
            scene.push_glass(GlassMarker {
                id: 0,
                region: tile(0, 0, 64, 64),
                composite: tile(16, 16, 48, 48),
            });
        }
        vibrant(&mut scene, red_blue_swap(), 0.5, CLIP);
        vibrant(&mut scene, grey(), 1.0, Rect::new(0.0, 0.0, 16.0, 64.0));
        scene
    };
    let (unmarked, marked) = (build(false), build(true));
    for aa in MODES {
        let stock = render(&mut renderer, aa, &unmarked);
        let glass = render_frame(&mut renderer, SIZE, SIZE, aa, &marked, Some(&mut NoGlass));
        // The identity criterion with content after the marker.
        assert!(max_delta(&stock, &glass) <= 1, "{aa:?}");
    }
    assert!(renderer.glass_pipelines_compiled());
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn a_frame_without_new_blends_is_stock() {
    if !Path::new("target/glass-baseline").is_dir() {
        eprintln!("glass.baseline.skipped: target/glass-baseline is absent");
        return;
    }
    let mut renderer = gpu_renderer(AaSupport::all());
    // A frame with its own matrix table first, so the stock frames after it must rebind the
    // identity table.
    let mut swapped = Scene::new();
    quadrants(&mut swapped);
    vibrant(&mut swapped, red_blue_swap(), 0.5, CLIP);
    for (aa, label) in [(AaConfig::Area, "area"), (AaConfig::Msaa16, "msaa16")] {
        render(&mut renderer, aa, &swapped);
        for (name, scene) in baseline_scenes() {
            let out = render_frame(
                &mut renderer,
                BASELINE_SIZE,
                BASELINE_SIZE,
                aa,
                &scene,
                None,
            );
            let path = baseline_path(name, label);
            let expected = image::open(&path)
                .unwrap_or_else(|e| panic!("missing baseline {}: {e}", path.display()))
                .to_rgba8()
                .into_raw();
            assert!(
                out == expected,
                "{name} with {label} differs from the baseline"
            );
        }
    }
}
