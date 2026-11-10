// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Advances the running glass phase and sets up its resume dispatch size.

// `glass_state` as fine binds it: [0] the running phase, [4 + j - 1] phase j's stop count.
@group(0) @binding(0)
var<storage, read_write> glass_state: array<atomic<u32>>;

// One (count_x, 1, 1, pad) dispatch per phase; a separate buffer, since a buffer can't be
// indirect arguments and read-write storage in one dispatch.
@group(0) @binding(1)
var<storage, read_write> glass_args: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let phase = atomicAdd(&glass_state[0], 1u) + 1u;
    let args = 4u * (phase - 1u);
    glass_args[args] = atomicLoad(&glass_state[4u + phase - 1u]);
    glass_args[args + 1u] = 1u;
    glass_args[args + 2u] = 1u;
    glass_args[args + 3u] = 0u;
}
