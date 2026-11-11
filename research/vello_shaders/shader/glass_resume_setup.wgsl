// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Advances the running glass phase and sets up its resume dispatch size; the `start`
// permutation sets up the phase 0 dispatch over list 0 instead. Each clamps its dispatch to
// the list's capacity and adds the excess to the overflow count.

#import config

@group(0) @binding(0)
var<uniform> config: Config;

// `glass_state` as fine binds it: [0] the running phase, [1] list 0's length, [2] the
// overflow count, [4 + j - 1] phase j's stop count.
@group(0) @binding(1)
var<storage, read_write> glass_state: array<atomic<u32>>;

// One (count_x, 1, 1, pad) dispatch per phase, then list 0's at slot `MAX_GLASS_PHASES`; a
// separate buffer, since a buffer can't be indirect arguments and read-write storage in one
// dispatch.
@group(0) @binding(2)
var<storage, read_write> glass_args: array<u32>;

// `GLASS_STATE_OVERFLOW` in `fine.wgsl`.
const GLASS_STATE_OVERFLOW = 2u;

// Writes a one-dimensional dispatch of `count` workgroups at `args`, at most `capacity`, and
// counts the workgroups left out. The CPU keeps every capacity within the device's
// `max_compute_workgroups_per_dimension`.
fn write_args(args: u32, count: u32, capacity: u32) {
    if count > capacity {
        atomicAdd(&glass_state[GLASS_STATE_OVERFLOW], count - capacity);
    }
    glass_args[args] = min(count, capacity);
    glass_args[args + 1u] = 1u;
    glass_args[args + 2u] = 1u;
    glass_args[args + 3u] = 0u;
}

#ifdef start
// `GLASS_STATE_TILES0` in `fine.wgsl`.
const GLASS_STATE_TILES0 = 1u;
// Four words per phase times `MAX_GLASS_PHASES` (4) in `vello_encoding`.
const LIST0_ARGS = 16u;

@compute @workgroup_size(1)
fn main() {
    write_args(LIST0_ARGS, atomicLoad(&glass_state[GLASS_STATE_TILES0]), config.glass_capacity_0);
}
#else
// Returns the most tiles that can stop for phase `j`, from 1; `glass_capacity` in `fine.wgsl`.
fn glass_capacity(j: u32) -> u32 {
    switch j {
        case 1u: { return config.glass_capacity_1; }
        case 2u: { return config.glass_capacity_2; }
        case 3u: { return config.glass_capacity_3; }
        case 4u: { return config.glass_capacity_4; }
        default: { return 0u; }
    }
}

@compute @workgroup_size(1)
fn main() {
    let phase = atomicAdd(&glass_state[0], 1u) + 1u;
    write_args(4u * (phase - 1u), atomicLoad(&glass_state[4u + phase - 1u]), glass_capacity(phase));
}
#endif
