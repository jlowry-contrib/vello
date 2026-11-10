// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glass markers: points in paint order where glass reads its backdrop.

pub use vello_encoding::{
    GlassMarker, GlassMarkers, GlassSkip, IRect, MAX_GLASS_LAYERS, MAX_GLASS_PHASES, assign_phases,
    pack_shelves,
};
