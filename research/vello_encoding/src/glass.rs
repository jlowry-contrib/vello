// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glass markers, phase assignment and atlas packing.

use crate::Transform;

/// The most glass phases a frame renders; deeper markers fold into the last one.
// The glass design's phase cap: it bounds the number of fine dispatches per frame.
pub const MAX_GLASS_PHASES: u32 = 4;

/// The most open layers a glass marker may sit under.
// The glass design's limit: coarse keeps a fixed eight-level stack of open layers.
pub const MAX_GLASS_LAYERS: u32 = 8;

/// Side of a fine tile in pixels, matching `TILE_WIDTH` and `TILE_HEIGHT` in `config.wgsl`.
pub(crate) const TILE: i32 = 16;

/// Set in a `GLASS` draw object's marker-index word when the frame leaves the marker out:
/// coarse then writes no glass command for it.
// Bit 31: marker indices count encoded markers, far below 2^31.
pub const GLASS_MARKER_SKIP: u32 = 1 << 31;

/// An integer rectangle in device pixels, half-open on its right and bottom edges.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct IRect {
    /// Left edge.
    pub x0: i32,
    /// Top edge.
    pub y0: i32,
    /// Right edge, exclusive.
    pub x1: i32,
    /// Bottom edge, exclusive.
    pub y1: i32,
}

impl IRect {
    /// Returns `true` if the rectangle covers no pixels.
    pub fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }

    /// Returns `true` if the two rectangles share at least one pixel.
    pub fn intersects(&self, other: &Self) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && self.x0 < other.x1
            && other.x0 < self.x1
            && self.y0 < other.y1
            && other.y0 < self.y1
    }

    /// Returns the pixels the two rectangles share, which may be empty.
    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            x0: self.x0.max(other.x0),
            y0: self.y0.max(other.y0),
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
        }
    }

    /// Returns `true` if `other` lies inside this rectangle.
    pub(crate) fn contains(self, other: Self) -> bool {
        self.x0 <= other.x0 && self.y0 <= other.y0 && other.x1 <= self.x1 && other.y1 <= self.y1
    }

    /// Grows the rectangle outward to multiples of `step`.
    ///
    /// An edge whose multiple lies outside `i32` saturates to the outermost multiple inside it.
    ///
    /// # Panics
    ///
    /// Panics if `step` is not positive.
    pub fn snap_out(self, step: i32) -> Self {
        assert!(step > 0, "snap step must be positive, got {step}");
        let step = i64::from(step);
        // The multiples of `step` nearest the `i32` limits, inside them.
        let lowest = -(-i64::from(i32::MIN)).div_euclid(step) * step;
        let highest = i64::from(i32::MAX).div_euclid(step) * step;
        let snap = |v: i64| v.clamp(lowest, highest) as i32;
        let down = |v: i32| snap(i64::from(v).div_euclid(step) * step);
        let up = |v: i32| snap(-(-i64::from(v)).div_euclid(step) * step);
        Self {
            x0: down(self.x0),
            y0: down(self.y0),
            x1: up(self.x1),
            y1: up(self.y1),
        }
    }

    /// Clamps every edge to `0..=width` horizontally and `0..=height` vertically.
    pub fn clamp(self, width: u32, height: u32) -> Self {
        let w = i32::try_from(width).unwrap_or(i32::MAX);
        let h = i32::try_from(height).unwrap_or(i32::MAX);
        Self {
            x0: self.x0.clamp(0, w),
            y0: self.y0.clamp(0, h),
            x1: self.x1.clamp(0, w),
            y1: self.y1.clamp(0, h),
        }
    }

    /// Moves the rectangle by `(dx, dy)`, saturating each edge at the `i32` limits.
    pub fn translate(self, dx: i32, dy: i32) -> Self {
        Self {
            x0: self.x0.saturating_add(dx),
            y0: self.y0.saturating_add(dy),
            x1: self.x1.saturating_add(dx),
            y1: self.y1.saturating_add(dy),
        }
    }
}

/// A point in paint order where glass samples its backdrop and composites.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GlassMarker {
    /// Caller-chosen identifier, passed through to the glass passes.
    pub id: u32,
    /// Pixels whose backdrop the glass reads.
    pub region: IRect,
    /// Pixels the glass writes; must lie inside `region`.
    pub composite: IRect,
}

impl GlassMarker {
    /// Snaps both rectangles outward to whole tiles.
    pub(crate) fn snap(self) -> Self {
        Self {
            id: self.id,
            region: self.region.snap_out(TILE),
            composite: self.composite.snap_out(TILE),
        }
    }
}

/// Why [`Encoding::encode_glass`](crate::Encoding::encode_glass) or
/// [`Encoding::append`](crate::Encoding::append) left markers out.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GlassSkip {
    /// The region or composite covers no pixels.
    Empty,
    /// The marker sits under more than [`MAX_GLASS_LAYERS`] open layers.
    TooDeep,
    /// The appended child's transform is not an integer translation.
    NonIntegerTransform,
}

/// The glass markers of an encoding, in paint order.
#[derive(Clone, Debug, Default)]
pub struct GlassMarkers {
    /// Markers with region and composite snapped outward to 16 px tiles.
    pub markers: Vec<GlassMarker>,
    /// Offset of each marker's `GLASS` draw data in the draw data stream.
    pub draw_data_offsets: Vec<usize>,
    /// Number of layers open at each marker.
    pub depths: Vec<u32>,
}

impl GlassMarkers {
    /// Returns the number of markers.
    pub fn len(&self) -> usize {
        self.markers.len()
    }

    /// Returns `true` if there are no markers.
    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.markers.clear();
        self.draw_data_offsets.clear();
        self.depths.clear();
    }

    pub(crate) fn push(&mut self, marker: GlassMarker, draw_data_offset: usize, depth: u32) {
        self.markers.push(marker);
        self.draw_data_offsets.push(draw_data_offset);
        self.depths.push(depth);
    }

    /// Appends `other`'s markers, rebasing their indices and offsets and moving them.
    ///
    /// `draw_data` must already hold `other`'s draw data at `draw_data_base`.
    ///
    /// # Errors
    ///
    /// Returns [`GlassSkip::NonIntegerTransform`] if `other` has markers and `transform` is not
    /// an integer translation. Their draw objects then carry [`GLASS_MARKER_SKIP`] and an empty
    /// composite, and none of them is added.
    pub(crate) fn append(
        &mut self,
        other: &Self,
        draw_data: &mut [u32],
        draw_data_base: usize,
        depth_base: u32,
        transform: Option<&Transform>,
    ) -> Result<(), GlassSkip> {
        if other.is_empty() {
            return Ok(());
        }
        let Some((dx, dy)) = transform.map_or(Some((0, 0)), integer_translation) else {
            for &offset in &other.draw_data_offsets {
                let offset = offset + draw_data_base;
                draw_data[offset..offset + 3].copy_from_slice(&[GLASS_MARKER_SKIP, 0, 0]);
            }
            return Err(GlassSkip::NonIntegerTransform);
        };
        let index_base = self.markers.len() as u32;
        for (i, marker) in other.markers.iter().enumerate() {
            let offset = other.draw_data_offsets[i] + draw_data_base;
            let moved = GlassMarker {
                id: marker.id,
                region: marker.region.translate(dx, dy),
                composite: marker.composite.translate(dx, dy),
            }
            .snap();
            let [min, max] = composite_tile_words(moved.composite);
            draw_data[offset] = index_base + i as u32;
            draw_data[offset + 1] = min;
            draw_data[offset + 2] = max;
            self.push(moved, offset, other.depths[i] + depth_base);
        }
        Ok(())
    }
}

/// Returns the translation of `transform`, if it is a translation by whole pixels.
fn integer_translation(transform: &Transform) -> Option<(i32, i32)> {
    let [tx, ty] = transform.translation;
    // 2^24 is where f32 stops representing every integer.
    let integral = |v: f32| v.fract() == 0.0 && v.abs() < (1 << 24) as f32;
    (transform.matrix == Transform::IDENTITY.matrix && integral(tx) && integral(ty))
        .then_some((tx as i32, ty as i32))
}

/// Packs a tile-aligned composite rectangle into its first and one-past-last tile words.
///
/// Each word is `x | y << 16`, with tile coordinates clamped to `0..=0xffff`.
pub(crate) fn composite_tile_words(composite: IRect) -> [u32; 2] {
    let pack = |x: i32, y: i32| {
        let x = (x / TILE).clamp(0, 0xffff) as u32;
        let y = (y / TILE).clamp(0, 0xffff) as u32;
        x | y << 16
    };
    [
        pack(composite.x0, composite.y0),
        pack(composite.x1, composite.y1),
    ]
}

/// Assigns each marker its glass phase, in paint order.
///
/// A marker's phase is one more than the highest phase of an earlier marker whose
/// composite meets its region, and 1 when there is none. A marker that would need a
/// phase past [`MAX_GLASS_PHASES`] gets phase 0: the frame leaves it out, so it
/// raises no later marker's phase. Returns the phases and how many markers the cap
/// left out.
pub fn assign_phases(markers: &[GlassMarker]) -> (Vec<u32>, usize) {
    assign_phases_fitting(markers, |_, _| true)
}

/// Assigns phases as [`assign_phases`] does, also leaving out each marker `fits` rejects.
///
/// `fits(i, phase)` is called in paint order for each marker within the phase cap. A
/// rejected marker gets phase 0 and raises no later marker's phase, and later markers are
/// still tried. Returns the phases and how many markers the cap left out.
pub fn assign_phases_fitting(
    markers: &[GlassMarker],
    fits: impl FnMut(usize, u32) -> bool,
) -> (Vec<u32>, usize) {
    phases_fitting(markers, false, fits)
}

/// Glass phases with markers past the cap folded into the last phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldedPhases {
    /// Each marker's phase, in paint order; 0 for a marker `fits` rejected.
    pub phases: Vec<u32>,
    /// How many kept markers were folded into [`MAX_GLASS_PHASES`].
    pub folded: usize,
}

/// Assigns phases as [`assign_phases_fitting`] does, folding markers past the cap into the
/// last phase instead of leaving them out.
///
/// A marker that would need a phase past [`MAX_GLASS_PHASES`] takes `MAX_GLASS_PHASES`, and
/// `fits(i, MAX_GLASS_PHASES)` decides it as for any marker. A kept folded marker counts in
/// [`FoldedPhases::folded`] and raises later markers' phases like any kept marker, so their
/// phases fold too. A folded marker's region may meet an earlier same-phase composite: its
/// backdrop there needs [`backdrop_fills`].
pub fn fold_phases_fitting(
    markers: &[GlassMarker],
    fits: impl FnMut(usize, u32) -> bool,
) -> FoldedPhases {
    let (phases, folded) = phases_fitting(markers, true, fits);
    FoldedPhases { phases, folded }
}

/// Assigns phases in paint order, folding or leaving out markers past the cap.
///
/// Returns the phases and how many markers passed the cap: those left out, or the kept
/// ones folded.
fn phases_fitting(
    markers: &[GlassMarker],
    fold: bool,
    mut fits: impl FnMut(usize, u32) -> bool,
) -> (Vec<u32>, usize) {
    let mut phases: Vec<u32> = Vec::with_capacity(markers.len());
    let mut past_cap = 0;
    for (i, marker) in markers.iter().enumerate() {
        let below = markers[..i]
            .iter()
            .zip(&phases)
            .filter(|(earlier, _)| earlier.composite.intersects(&marker.region))
            .map(|(_, &phase)| phase)
            .max()
            .unwrap_or(0);
        let phase = below + 1;
        if phase > MAX_GLASS_PHASES && !fold {
            past_cap += 1;
            phases.push(0);
        } else if fits(i, phase.min(MAX_GLASS_PHASES)) {
            if phase > MAX_GLASS_PHASES {
                past_cap += 1;
            }
            phases.push(phase.min(MAX_GLASS_PHASES));
        } else {
            phases.push(0);
        }
    }
    (phases, past_cap)
}

/// A backdrop copy a folded glass marker needs before its phase's glass passes record.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BackdropFill {
    /// The phase both markers share.
    pub phase: u32,
    /// Index of the earlier marker whose backdrop slot is read.
    pub from: usize,
    /// Index of the later marker whose backdrop slot is written.
    pub to: usize,
    /// Pixels to copy: `from`'s composite within `to`'s region.
    ///
    /// Tile-aligned, except where a marker was clamped at the target's right or bottom edge.
    pub rect: IRect,
}

/// Lists the backdrop copies that markers sharing a phase need, in paint order of `to`.
///
/// On a tile of marker `to`'s region that also lies in an earlier same-phase marker `from`'s
/// composite, fine stops at `from` and resumes only in that phase's fine pass, after its glass
/// passes ran, so `to`'s backdrop is never snapshot there. Same-phase composites never
/// overlap and a composite lies inside its region, so `from`'s slot holds the backdrop
/// `from` saw on those tiles, which the renderer copies into `to`'s slot. Only markers
/// folded by [`fold_phases_fitting`] can meet a same-phase composite, so other frames get
/// no fills. Markers with phase 0 are skipped.
pub fn backdrop_fills(markers: &[GlassMarker], phases: &[u32]) -> Vec<BackdropFill> {
    let mut fills = Vec::new();
    for (to, (marker, &phase)) in markers.iter().zip(phases).enumerate() {
        if phase == 0 {
            continue;
        }
        for (from, earlier) in markers[..to].iter().enumerate() {
            if phases[from] == phase && earlier.composite.intersects(&marker.region) {
                fills.push(BackdropFill {
                    phase,
                    from,
                    to,
                    rect: earlier.composite.intersect(&marker.region),
                });
            }
        }
    }
    fills
}

/// Packs `(width, height)` slots onto shelves, returning their origins and the atlas size.
///
/// Slots go on shelves tallest first, left to right. The atlas width is a power of two from
/// the larger of `min_width` and the widest slot rounded up, widened while that saves area
/// and stays within `max_side`; the height is rounded up to a power of two. Among the
/// widths whose atlas fits in `max_side`, the smallest area wins, then the smaller longer
/// side, then the narrower. When no width fits, the tried atlas with the shortest longer side
/// is returned, so the caller can see it passes `max_side`.
pub fn pack_shelves(
    sizes: &[(u32, u32)],
    min_width: u32,
    max_side: u32,
) -> (Vec<(u32, u32)>, (u32, u32)) {
    let widest = sizes.iter().map(|&(w, _)| w).max().unwrap_or(0);
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(sizes[i].1));
    let start = min_width.max(widest.next_power_of_two());
    let (mut best_key, mut best) = (None, (Vec::new(), (0, 0)));
    let mut width = start;
    loop {
        let (origins, size) = pack_shelves_at(sizes, &order, width);
        let long = size.0.max(size.1);
        let fits = long <= max_side;
        let area = u64::from(size.0) * u64::from(size.1);
        // Fitting atlases sort first, by area; others by their longer side.
        let key = if fits {
            (false, area, long, width)
        } else {
            (true, u64::from(long), long, width)
        };
        if best_key.is_none_or(|best_key| key < best_key) {
            (best_key, best) = (Some(key), (origins, size));
        }
        // A wider atlas can't fit past `max_side`, nor get shorter once one shelf holds every
        // slot.
        let one_shelf = sizes.iter().map(|&(w, _)| u64::from(w)).sum::<u64>() <= u64::from(width);
        match width.checked_mul(2) {
            Some(wider) if width < max_side && !one_shelf => width = wider,
            _ => break,
        }
    }
    best
}

/// Packs slots in `order` onto shelves of `width`, as [`pack_shelves`] does for one width.
fn pack_shelves_at(
    sizes: &[(u32, u32)],
    order: &[usize],
    width: u32,
) -> (Vec<(u32, u32)>, (u32, u32)) {
    let mut origins = vec![(0, 0); sizes.len()];
    let (mut x, mut y, mut shelf) = (0, 0, 0);
    for &i in order {
        let (w, h) = sizes[i];
        if x + w > width {
            y += shelf;
            x = 0;
            shelf = 0;
        }
        origins[i] = (x, y);
        x += w;
        shelf = shelf.max(h);
    }
    (origins, (width, (y + shelf).next_power_of_two()))
}

#[cfg(test)]
mod tests {
    use super::{
        GLASS_MARKER_SKIP, GlassMarker, GlassSkip, IRect, TILE, assign_phases,
        assign_phases_fitting, backdrop_fills, fold_phases_fitting, pack_shelves,
    };
    use crate::{DrawTag, Encoding, Transform};

    fn r((x0, y0, x1, y1): (i32, i32, i32, i32)) -> IRect {
        IRect { x0, y0, x1, y1 }
    }

    fn m(region: (i32, i32, i32, i32), composite: (i32, i32, i32, i32)) -> GlassMarker {
        GlassMarker {
            id: 0,
            region: r(region),
            composite: r(composite),
        }
    }

    fn phases(markers: &[GlassMarker]) -> Vec<u32> {
        let snapped: Vec<_> = markers
            .iter()
            .map(|m| GlassMarker {
                id: m.id,
                region: m.region.snap_out(TILE),
                composite: m.composite.snap_out(TILE),
            })
            .collect();
        assign_phases(&snapped).0
    }

    /// `n` snapped markers in a diagonal row, each composite meeting only the next region.
    fn chain(n: i32) -> Vec<GlassMarker> {
        (0..n)
            .map(|i| {
                let o = 64 * i;
                m(
                    (64 + o, 64 + o, 192 + o, 192 + o),
                    (96 + o, 96 + o, 160 + o, 160 + o),
                )
                .snap()
            })
            .collect()
    }

    fn glass_words(encoding: &Encoding, index: usize) -> [u32; 3] {
        let offset = encoding.glass.draw_data_offsets[index];
        encoding.draw_data[offset..offset + 3].try_into().unwrap()
    }

    #[test]
    fn disjoint_markers_share_phase_one() {
        let a = m((0, 0, 64, 64), (16, 16, 48, 48));
        let b = m((128, 0, 192, 64), (144, 16, 176, 48));
        assert_eq!(phases(&[a, b]), [1, 1]);
    }

    #[test]
    fn marker_over_an_earlier_composite_takes_the_next_phase() {
        let a = m((0, 0, 64, 64), (16, 16, 48, 48));
        let b = m((32, 32, 96, 96), (48, 48, 80, 80));
        assert_eq!(phases(&[a, b]), [1, 2]);
    }

    #[test]
    fn marker_touching_only_an_earlier_region_stays_in_its_phase() {
        let a = m((0, 0, 64, 64), (16, 16, 48, 48));
        let b = m((64, 0, 128, 64), (80, 16, 112, 48));
        assert_eq!(phases(&[a, b]), [1, 1]);
    }

    #[test]
    fn rectangles_that_meet_only_after_snapping_take_the_next_phase() {
        // Pixel rectangles don't meet, but m1's region snaps to x = 32 and m0's composite
        // covers tile column 2.
        let a = m((0, 0, 64, 64), (16, 16, 40, 40));
        let b = m((44, 0, 128, 64), (64, 16, 112, 48));
        assert!(!a.composite.intersects(&b.region));
        assert_eq!(phases(&[a, b]), [1, 2]);
    }

    #[test]
    fn markers_past_the_cap_are_left_out() {
        let markers: Vec<_> = (0..6)
            .map(|i| {
                m(
                    (32 * i, 0, 32 * i + 64, 64),
                    (32 * i + 16, 16, 32 * i + 48, 48),
                )
            })
            .collect();
        let snapped: Vec<_> = markers
            .iter()
            .map(|m| GlassMarker {
                id: m.id,
                region: m.region.snap_out(TILE),
                composite: m.composite.snap_out(TILE),
            })
            .collect();
        let (phases, capped) = assign_phases(&snapped);
        // The fifth is left out, so the sixth, which meets only its composite, starts over.
        assert_eq!(phases, [1, 2, 3, 4, 0, 1]);
        assert_eq!(capped, 1);
    }

    #[test]
    fn rejected_markers_raise_no_later_phase() {
        let a = m((0, 0, 64, 64), (16, 16, 48, 48));
        let b = m((32, 32, 96, 96), (48, 48, 80, 80));
        let c = m((64, 64, 128, 128), (80, 80, 112, 112));
        let snapped: Vec<_> = [a, b, c].iter().map(|m| m.snap()).collect();
        let mut tried = Vec::new();
        let (phases, capped) = assign_phases_fitting(&snapped, |i, phase| {
            tried.push((i, phase));
            i != 1
        });
        // b is rejected, so c, which meets only b's composite, takes phase 1.
        assert_eq!(phases, [1, 0, 1]);
        assert_eq!(capped, 0);
        assert_eq!(tried, [(0, 1), (1, 2), (2, 1)]);
    }

    #[test]
    fn eight_marker_chain_folds_the_last_four() {
        let markers = chain(8);
        let folded = fold_phases_fitting(&markers, |_, _| true);
        assert_eq!(folded.phases, [1, 2, 3, 4, 4, 4, 4, 4]);
        assert_eq!(folded.folded, 4);
    }

    #[test]
    fn a_rejected_marker_still_raises_nothing_when_folding() {
        let a = m((0, 0, 64, 64), (16, 16, 48, 48));
        let b = m((32, 32, 96, 96), (48, 48, 80, 80));
        let c = m((64, 64, 128, 128), (80, 80, 112, 112));
        let snapped: Vec<_> = [a, b, c].iter().map(|m| m.snap()).collect();
        // b is rejected, so c, which meets only b's composite, takes phase 1.
        let folded = fold_phases_fitting(&snapped, |i, _| i != 1);
        assert_eq!(folded.phases, [1, 0, 1]);
        assert_eq!(folded.folded, 0);
    }

    #[test]
    fn fills_cover_each_same_phase_composite_in_a_folded_region() {
        let markers = chain(6);
        let phases = fold_phases_fitting(&markers, |_, _| true).phases;
        let fills = backdrop_fills(&markers, &phases);
        assert_eq!(
            fills.iter().map(|f| (f.from, f.to)).collect::<Vec<_>>(),
            [(3, 4), (4, 5)]
        );
        for f in &fills {
            assert_eq!(f.phase, 4);
            assert_eq!(
                f.rect,
                markers[f.from].composite.intersect(&markers[f.to].region)
            );
            assert!(!f.rect.is_empty());
        }
    }

    #[test]
    fn a_folded_marker_rejected_by_fits_raises_nothing() {
        let folded = fold_phases_fitting(&chain(6), |i, _| i != 4);
        // Marker 4 would fold but is rejected, so marker 5, which meets only its composite,
        // takes phase 1.
        assert_eq!(folded.phases, [1, 2, 3, 4, 0, 1]);
        assert_eq!(folded.folded, 0);
    }

    #[test]
    fn unfolded_frames_need_no_fill() {
        let markers = chain(4);
        let phases = fold_phases_fitting(&markers, |_, _| true).phases;
        assert!(backdrop_fills(&markers, &phases).is_empty());
    }

    #[test]
    fn assign_phases_is_unchanged() {
        assert_eq!(assign_phases(&chain(5)), (vec![1, 2, 3, 4, 0], 1));
    }

    #[test]
    fn snapping_rounds_outward_to_tiles() {
        assert_eq!(
            r((5, 5, 20, 33)).snap_out(16),
            IRect {
                x0: 0,
                y0: 0,
                x1: 32,
                y1: 48
            }
        );
        assert_eq!(r((-5, 0, 16, 16)).snap_out(16).x0, -16);
        assert_eq!(r((-20, -33, -5, -1)).snap_out(16), r((-32, -48, 0, 0)));
    }

    #[test]
    fn clamp_and_translate_move_each_edge() {
        assert_eq!(r((-8, -8, 600, 40)).clamp(512, 32), r((0, 0, 512, 32)));
        assert_eq!(r((0, 0, 16, 16)).translate(3, -4), r((3, -4, 19, 12)));
        assert!(r((10, 10, 10, 40)).is_empty());
        assert!(!r((0, 0, 16, 16)).intersects(&r((16, 0, 32, 16))));
        assert!(r((0, 0, 17, 16)).intersects(&r((16, 0, 32, 16))));
    }

    #[test]
    fn push_glass_encodes_the_region_path_and_tag() {
        let mut encoding = Encoding::new();
        let marker = m((5, 5, 40, 40), (16, 16, 32, 32));
        assert_eq!(encoding.encode_glass(&marker), Ok(()));
        assert!(encoding.draw_tags.last() == Some(&DrawTag::GLASS));
        assert_eq!(encoding.draw_data, [0, 1 | 1 << 16, 2 | 2 << 16]);
        assert_eq!(encoding.n_paths, 1);
        assert_eq!(encoding.transforms, [Transform::IDENTITY]);
        assert_eq!(encoding.glass.markers.len(), 1);
        assert_eq!(encoding.glass.markers[0].region, r((0, 0, 48, 48)));
        assert_eq!(encoding.glass.markers[0].composite, r((16, 16, 32, 32)));
        assert_eq!(encoding.glass.draw_data_offsets, [0]);
        assert_eq!(encoding.glass.depths, [0]);
    }

    #[test]
    fn glass_tag_carries_three_scene_words_and_three_info_words() {
        assert_eq!(DrawTag::GLASS.info_size(), 3);
        assert_eq!((DrawTag::GLASS.0 >> 2) & 0x7, 3);
        assert_eq!(DrawTag::GLASS.0 & 1, 0);
    }

    #[test]
    fn negative_composite_tiles_clamp_to_zero_before_packing() {
        let mut encoding = Encoding::new();
        let marker = m((-64, -32, 40, 40), (-48, -16, 24, 24));
        assert_eq!(encoding.encode_glass(&marker), Ok(()));
        assert_eq!(glass_words(&encoding, 0), [0, 0, 2 | 2 << 16]);
    }

    #[test]
    fn empty_marker_is_not_encoded() {
        let mut encoding = Encoding::new();
        let marker = m((10, 10, 10, 40), (10, 10, 10, 40));
        assert_eq!(encoding.encode_glass(&marker), Err(GlassSkip::Empty));
        let marker = m((0, 0, 32, 32), (16, 16, 16, 32));
        assert_eq!(encoding.encode_glass(&marker), Err(GlassSkip::Empty));
        assert!(encoding.is_empty());
        assert!(encoding.draw_tags.is_empty());
        assert!(encoding.glass.markers.is_empty());
    }

    #[test]
    fn marker_under_nine_open_layers_is_not_encoded() {
        let mut encoding = Encoding::new();
        encoding.n_open_clips = 9;
        let marker = m((0, 0, 32, 32), (0, 0, 16, 16));
        assert_eq!(encoding.encode_glass(&marker), Err(GlassSkip::TooDeep));
        assert!(encoding.draw_tags.is_empty());
        assert!(encoding.glass.markers.is_empty());
        encoding.n_open_clips = 8;
        assert_eq!(encoding.encode_glass(&marker), Ok(()));
        assert_eq!(encoding.glass.depths, [8]);
    }

    #[test]
    #[should_panic(expected = "glass marker composite must lie inside its region")]
    fn composite_outside_region_panics() {
        let mut encoding = Encoding::new();
        let marker = m((0, 0, 32, 32), (16, 16, 48, 32));
        let _ = encoding.encode_glass(&marker);
    }

    #[test]
    fn append_rebases_and_translates_markers() {
        let mut a = Encoding::new();
        a.encode_glass(&m((0, 0, 64, 64), (16, 16, 48, 48)))
            .unwrap();
        let mut b = Encoding::new();
        b.encode_glass(&m((0, 0, 32, 32), (0, 0, 16, 16))).unwrap();
        b.encode_glass(&m((5, 5, 40, 40), (16, 16, 32, 32)))
            .unwrap();
        let translate = Transform {
            matrix: [1.0, 0.0, 0.0, 1.0],
            translation: [16.0, 32.0],
        };
        a.append(&b, &Some(translate)).unwrap();
        assert_eq!(a.glass.markers.len(), 3);
        let words: Vec<_> = (0..3).map(|i| glass_words(&a, i)[0]).collect();
        assert_eq!(words, [0, 1, 2]);
        assert_eq!(a.glass.markers[1].region, r((16, 32, 48, 64)));
        assert_eq!(a.glass.markers[1].composite, r((16, 32, 32, 48)));
        assert_eq!(a.glass.markers[2].region, r((16, 32, 64, 80)));
        assert_eq!(a.glass.markers[2].composite, r((32, 48, 48, 64)));
        assert_eq!(glass_words(&a, 1)[1..], [1 | 2 << 16, 2 | 3 << 16]);
        assert_eq!(glass_words(&a, 2)[1..], [2 | 3 << 16, 3 | 4 << 16]);
    }

    #[test]
    fn append_resnaps_markers_moved_by_part_of_a_tile() {
        let mut a = Encoding::new();
        let mut b = Encoding::new();
        b.encode_glass(&m((0, 0, 32, 32), (0, 0, 16, 16))).unwrap();
        let translate = Transform {
            matrix: [1.0, 0.0, 0.0, 1.0],
            translation: [8.0, -8.0],
        };
        a.append(&b, &Some(translate)).unwrap();
        assert_eq!(a.glass.markers[0].region, r((0, -16, 48, 32)));
        assert_eq!(a.glass.markers[0].composite, r((0, -16, 32, 16)));
        assert_eq!(glass_words(&a, 0), [0, 0, 2 | 1 << 16]);
    }

    #[test]
    fn append_adds_the_outer_open_depth() {
        let mut a = Encoding::new();
        a.n_open_clips = 3;
        let mut b = Encoding::new();
        b.n_open_clips = 6;
        b.encode_glass(&m((0, 0, 32, 32), (0, 0, 16, 16))).unwrap();
        a.append(&b, &None).unwrap();
        assert_eq!(a.glass.depths, [9]);
        assert_eq!(a.glass.markers[0].region, r((0, 0, 32, 32)));
    }

    #[test]
    fn append_under_a_fractional_translation_leaves_markers_out() {
        let fractional = Transform {
            matrix: [1.0, 0.0, 0.0, 1.0],
            translation: [0.5, 0.0],
        };
        let scale = Transform {
            matrix: [2.0, 0.0, 0.0, 2.0],
            translation: [0.0, 0.0],
        };
        for transform in [fractional, scale] {
            let mut a = Encoding::new();
            a.encode_glass(&m((0, 0, 32, 32), (0, 0, 16, 16))).unwrap();
            let mut b = Encoding::new();
            b.encode_glass(&m((0, 0, 32, 32), (0, 0, 16, 16))).unwrap();
            b.encode_glass(&m((32, 0, 64, 32), (32, 0, 48, 16)))
                .unwrap();
            let result = a.append(&b, &Some(transform));
            assert_eq!(result, Err(GlassSkip::NonIntegerTransform));
            // The rest of the child is appended; only its markers are left out.
            assert_eq!(a.draw_tags.len(), 3);
            assert_eq!(a.glass.markers.len(), 1);
            assert_eq!(a.glass.draw_data_offsets, [0]);
            assert_eq!(glass_words(&a, 0), [0, 0, 1 | 1 << 16]);
            assert_eq!(
                a.draw_data[3..],
                [GLASS_MARKER_SKIP, 0, 0, GLASS_MARKER_SKIP, 0, 0]
            );
        }
    }

    #[test]
    fn append_under_an_integer_translation_keeps_markers() {
        let mut a = Encoding::new();
        let mut b = Encoding::new();
        b.encode_glass(&m((0, 0, 32, 32), (0, 0, 16, 16))).unwrap();
        let translate = Transform {
            matrix: [1.0, 0.0, 0.0, 1.0],
            translation: [-16.0, 48.0],
        };
        assert_eq!(a.append(&b, &Some(translate)), Ok(()));
        assert_eq!(a.glass.markers.len(), 1);
        assert_eq!(a.glass.markers[0].region, r((-16, 48, 16, 80)));
        assert_eq!(glass_words(&a, 0)[0], 0);
    }

    #[test]
    fn snap_out_saturates_at_the_i32_limits() {
        let huge = r((i32::MIN, i32::MIN + 1, i32::MAX, i32::MAX - 1)).snap_out(16);
        assert_eq!(huge, r((i32::MIN, i32::MIN, i32::MAX - 15, i32::MAX - 15)));
        assert_eq!(r((i32::MIN + 1, 0, 16, 16)).snap_out(TILE).x0, i32::MIN);
        assert_eq!(
            r((0, 0, i32::MAX - 20, 16)).snap_out(TILE).x1,
            i32::MAX - 15
        );
        // Steps that don't divide 2^31 saturate inside the limits too.
        let odd = r((i32::MIN, 0, i32::MAX, 3)).snap_out(3);
        assert_eq!(odd, r((i32::MIN + 2, 0, i32::MAX - 1, 3)));
        let moved = r((0, 0, 16, 16)).translate(i32::MAX, i32::MIN);
        assert_eq!(moved, r((i32::MAX, i32::MIN, i32::MAX, i32::MIN + 16)));
        let back = r((i32::MIN, i32::MIN, 0, 0)).translate(-1, -1);
        assert_eq!(back, r((i32::MIN, i32::MIN, -1, -1)));
    }

    #[test]
    fn reset_clears_the_markers() {
        let mut encoding = Encoding::new();
        encoding
            .encode_glass(&m((0, 0, 32, 32), (0, 0, 16, 16)))
            .unwrap();
        encoding.reset();
        assert!(encoding.glass.markers.is_empty());
        assert!(encoding.glass.draw_data_offsets.is_empty());
        assert!(encoding.glass.depths.is_empty());
    }

    #[test]
    fn shelves_pack_without_overlap_into_power_of_two_sizes() {
        let sizes = [(320, 128), (64, 64), (512, 32)];
        let (origins, size) = pack_shelves(&sizes, 512, 8192);
        assert_eq!(size, (512, 256));
        let rects: Vec<_> = origins
            .iter()
            .zip(sizes)
            .map(|(&(x, y), (w, h))| (x, y, x + w, y + h))
            .collect();
        for (i, a) in rects.iter().enumerate() {
            assert!(a.2 <= size.0 && a.3 <= size.1, "slot {i} leaves the atlas");
            for b in &rects[i + 1..] {
                let disjoint = a.2 <= b.0 || b.2 <= a.0 || a.3 <= b.1 || b.3 <= a.1;
                assert!(disjoint, "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn shelves_widen_before_they_grow_tall() {
        // Twenty 416 px slots, one per shelf at 512 px, would stack to 8320 px and round up to
        // 16384; wider shelves fit them in an 8192 px limit.
        let sizes = [(416, 416); 20];
        let (origins, size) = pack_shelves(&sizes, 512, 8192);
        assert_eq!(size, (2048, 4096));
        assert!(
            origins
                .iter()
                .all(|&(x, y)| x + 416 <= 2048 && y + 416 <= 4096)
        );
        // About 256 px slots, which stacked to 8192 px, now take a quarter of that height.
        let (_, size) = pack_shelves(&[(272, 272); 20], 512, 8192);
        assert_eq!(size, (1024, 2048));
    }

    #[test]
    fn shelves_report_the_shortest_atlas_when_none_fits() {
        let (_, size) = pack_shelves(&[(416, 416); 20], 512, 1024);
        assert_eq!(size, (1024, 8192));
    }

    #[test]
    fn shelves_widen_to_the_widest_slot() {
        let (_, size) = pack_shelves(&[(700, 10)], 512, 8192);
        assert_eq!(size, (1024, 16));
    }
}
