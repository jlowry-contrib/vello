// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vibrant colour matrices, for layers that recolour their backdrop.

use crate::DrawBeginClip;

/// The most distinct vibrant matrices one encoding holds.
// The glass plan's limit of 255 per frame, which an 8-bit index in the blend word holds.
pub const MAX_VIBRANT_MATRICES: usize = 255;

/// Words of one matrix in the GPU table: four rows of four weights, the offsets, then the
/// clamp bounds and two words of padding. Mirrors `VibrantMatrix` in `fine.wgsl`.
pub const VIBRANT_MATRIX_WORDS: usize = 24;

/// A 4×5 colour matrix that a vibrant layer applies to its un-premultiplied backdrop.
///
/// Row `i` maps `[r, g, b, a, 1]` to output channel `i` (red, green, blue, alpha). Each
/// output is clamped to `clamp`, then to `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VibrantMatrix {
    /// The output rows: four input weights and an offset each.
    pub rows: [[f32; 5]; 4],
    /// The lower and upper bound of every output channel, applied before `[0, 1]`.
    pub clamp: (f32, f32),
}

impl VibrantMatrix {
    /// The matrix that leaves the backdrop unchanged.
    pub const IDENTITY: Self = Self {
        rows: [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        clamp: (0.0, 1.0),
    };

    /// Returns the matrix as the bits of its floats, for deduplication.
    fn bits(&self) -> [u32; 22] {
        let mut bits = [0; 22];
        for (i, value) in self.rows.iter().flatten().enumerate() {
            bits[i] = value.to_bits();
        }
        bits[20] = self.clamp.0.to_bits();
        bits[21] = self.clamp.1.to_bits();
        bits
    }

    /// Returns the matrix in its GPU layout.
    pub fn gpu_words(&self) -> [f32; VIBRANT_MATRIX_WORDS] {
        let mut words = [0.0; VIBRANT_MATRIX_WORDS];
        for (i, row) in self.rows.iter().enumerate() {
            words[4 * i..4 * i + 4].copy_from_slice(&row[..4]);
            words[16 + i] = row[4];
        }
        words[20] = self.clamp.0;
        words[21] = self.clamp.1;
        words
    }
}

impl Default for VibrantMatrix {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// The distinct vibrant matrices of an encoding and the layers that use them.
#[derive(Clone, Debug, Default)]
pub struct VibrantMatrices {
    matrices: Vec<VibrantMatrix>,
    /// The draw data offset of each vibrant `BEGIN_CLIP`'s blend word.
    draw_data_offsets: Vec<usize>,
}

impl VibrantMatrices {
    /// Returns the matrices, in table order.
    pub fn matrices(&self) -> &[VibrantMatrix] {
        &self.matrices
    }

    /// Returns the number of distinct matrices.
    pub fn len(&self) -> usize {
        self.matrices.len()
    }

    /// Returns `true` if there are no matrices.
    pub fn is_empty(&self) -> bool {
        self.matrices.is_empty()
    }

    /// Returns the table the fine shaders bind: each matrix in its GPU layout.
    pub fn gpu_table(&self) -> Vec<f32> {
        self.matrices.iter().flat_map(|m| m.gpu_words()).collect()
    }

    /// Returns how many of `other`'s matrices this table has no room for.
    pub fn overflow(&self, other: &Self) -> usize {
        let new = other
            .matrices
            .iter()
            .filter(|m| self.find(m).is_none())
            .count();
        new.saturating_sub(MAX_VIBRANT_MATRICES - self.matrices.len())
    }

    pub(crate) fn clear(&mut self) {
        self.matrices.clear();
        self.draw_data_offsets.clear();
    }

    fn find(&self, matrix: &VibrantMatrix) -> Option<u32> {
        let bits = matrix.bits();
        self.matrices
            .iter()
            .position(|m| m.bits() == bits)
            .map(|i| i as u32)
    }

    /// Returns `matrix`'s index, adding it if it is new, or `None` if the table is full.
    pub(crate) fn index(&mut self, matrix: &VibrantMatrix) -> Option<u32> {
        self.find(matrix).or_else(|| {
            (self.matrices.len() < MAX_VIBRANT_MATRICES).then(|| {
                self.matrices.push(*matrix);
                self.matrices.len() as u32 - 1
            })
        })
    }

    /// Records a vibrant `BEGIN_CLIP` whose blend word sits at `draw_data_offset`.
    pub(crate) fn push_layer(&mut self, draw_data_offset: usize) {
        self.draw_data_offsets.push(draw_data_offset);
    }

    /// Merges `other`'s matrices into this table and rewrites the index in each of its
    /// layers' blend words.
    ///
    /// `draw_data` must already hold `other`'s draw data at `draw_data_base`. A layer whose
    /// matrix doesn't fit falls back to normal blending.
    pub(crate) fn append(&mut self, other: &Self, draw_data: &mut [u32], draw_data_base: usize) {
        if other.matrices.is_empty() {
            return;
        }
        let remap: Vec<Option<u32>> = other.matrices.iter().map(|m| self.index(m)).collect();
        for &offset in &other.draw_data_offsets {
            let offset = offset + draw_data_base;
            let word = draw_data[offset];
            let old = (word >> DrawBeginClip::VIBRANT_INDEX_SHIFT) & 0xff;
            match remap[old as usize] {
                Some(index) => {
                    draw_data[offset] = DrawBeginClip::vibrant_blend_mode(index);
                    self.draw_data_offsets.push(offset);
                }
                None => draw_data[offset] = DrawBeginClip::NORMAL_BLEND_MODE,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_VIBRANT_MATRICES, VibrantMatrix};
    use crate::{DrawBeginClip, Encoding};

    /// A made-up matrix that scales red by `k`.
    fn scale_red(k: f32) -> VibrantMatrix {
        let mut m = VibrantMatrix::IDENTITY;
        m.rows[0][0] = k;
        m
    }

    /// Returns the blend words of an encoding that holds only `BEGIN_CLIP`s, two words each.
    fn blend_words(encoding: &Encoding) -> Vec<u32> {
        encoding.draw_data.chunks(2).map(|c| c[0]).collect()
    }

    fn index_of(word: u32) -> u32 {
        assert_ne!(word & DrawBeginClip::VIBRANT_BLEND_MODE, 0, "not vibrant");
        (word >> DrawBeginClip::VIBRANT_INDEX_SHIFT) & 0xff
    }

    #[test]
    fn equal_matrices_share_an_index() {
        let mut encoding = Encoding::new();
        assert!(encoding.encode_vibrant_begin_clip(&scale_red(2.0), 1.0));
        assert!(encoding.encode_vibrant_begin_clip(&scale_red(3.0), 1.0));
        assert!(encoding.encode_vibrant_begin_clip(&scale_red(2.0), 1.0));
        assert_eq!(encoding.vibrant.len(), 2);
        let indices: Vec<_> = blend_words(&encoding).into_iter().map(index_of).collect();
        assert_eq!(indices, [0, 1, 0]);
    }

    #[test]
    fn append_remaps_the_child_indices() {
        let (a, b) = (scale_red(2.0), scale_red(3.0));
        let mut parent = Encoding::new();
        parent.encode_vibrant_begin_clip(&a, 1.0);
        let mut child = Encoding::new();
        child.encode_vibrant_begin_clip(&b, 1.0);
        child.encode_vibrant_begin_clip(&a, 1.0);
        parent.append(&child, &None).unwrap();
        assert_eq!(parent.vibrant.matrices(), [a, b]);
        let indices: Vec<_> = blend_words(&parent).into_iter().map(index_of).collect();
        assert_eq!(indices, [0, 1, 0]);
        // The rewritten layers are tracked, so a further append into another scene remaps them.
        let mut grand = Encoding::new();
        grand.encode_vibrant_begin_clip(&b, 1.0);
        grand.append(&parent, &None).unwrap();
        let indices: Vec<_> = blend_words(&grand).into_iter().map(index_of).collect();
        assert_eq!(indices, [0, 1, 0, 1]);
    }

    #[test]
    fn a_full_table_falls_back_to_normal_blending() {
        let mut encoding = Encoding::new();
        for i in 0..MAX_VIBRANT_MATRICES {
            assert!(encoding.encode_vibrant_begin_clip(&scale_red(i as f32), 1.0));
        }
        assert!(!encoding.encode_vibrant_begin_clip(&scale_red(-1.0), 0.5));
        assert_eq!(
            *blend_words(&encoding).last().unwrap(),
            DrawBeginClip::NORMAL_BLEND_MODE
        );
        // A matrix already in the table still fits.
        assert!(encoding.encode_vibrant_begin_clip(&scale_red(7.0), 1.0));

        let mut child = Encoding::new();
        child.encode_vibrant_begin_clip(&scale_red(3.0), 1.0);
        child.encode_vibrant_begin_clip(&scale_red(-2.0), 1.0);
        assert_eq!(encoding.vibrant.overflow(&child.vibrant), 1);
        encoding.append(&child, &None).unwrap();
        let words = blend_words(&encoding);
        assert_eq!(index_of(words[words.len() - 2]), 3);
        assert_eq!(words[words.len() - 1], DrawBeginClip::NORMAL_BLEND_MODE);
    }

    #[test]
    fn gpu_words_put_weights_then_offsets_then_bounds() {
        let mut m = VibrantMatrix::IDENTITY;
        m.rows[2][4] = 0.25;
        m.clamp = (-0.5, 2.0);
        let w = m.gpu_words();
        assert_eq!(&w[8..12], &[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(&w[16..20], &[0.0, 0.0, 0.25, 0.0]);
        assert_eq!(&w[20..], &[-0.5, 2.0, 0.0, 0.0]);
    }
}
