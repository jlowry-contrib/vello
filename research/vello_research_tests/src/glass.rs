// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Deterministic scenes and helpers for the glass tests.

use vello::Scene;
use vello::kurbo::{Affine, Rect};
use vello::peniko::{Color, Fill, Gradient};

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
