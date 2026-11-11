// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Advances the running glass phase and sets up its resume dispatch size; the `start`
// permutation sets up the phase 0 dispatch over list 0 instead.

// `glass_state` as fine binds it: [0] the running phase, [1] list 0's length,
// [4 + j - 1] phase j's stop count.
@group(0) @binding(0)
var<storage, read_write> glass_state: array<atomic<u32>>;

// One (count_x, 1, 1, pad) dispatch per phase, then list 0's at slot `MAX_GLASS_PHASES`; a
// separate buffer, since a buffer can't be indirect arguments and read-write storage in one
// dispatch.
@group(0) @binding(1)
var<storage, read_write> glass_args: array<u32>;

#ifdef start
// `GLASS_STATE_TILES0` in `fine.wgsl`.
const GLASS_STATE_TILES0 = 1u;
// Four words per phase times `MAX_GLASS_PHASES` (4) in `vello_encoding`.
const LIST0_ARGS = 16u;

@compute @workgroup_size(1)
fn main() {
    glass_args[LIST0_ARGS] = atomicLoad(&glass_state[GLASS_STATE_TILES0]);
    glass_args[LIST0_ARGS + 1u] = 1u;
    glass_args[LIST0_ARGS + 2u] = 1u;
    glass_args[LIST0_ARGS + 3u] = 0u;
}
#else
@compute @workgroup_size(1)
fn main() {
    let phase = atomicAdd(&glass_state[0], 1u) + 1u;
    let args = 4u * (phase - 1u);
    glass_args[args] = atomicLoad(&glass_state[4u + phase - 1u]);
    glass_args[args + 1u] = 1u;
    glass_args[args + 2u] = 1u;
    glass_args[args + 3u] = 0u;
}
#endif
