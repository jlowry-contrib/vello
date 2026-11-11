// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renders from the upstream code, saved so later changes can prove the stock path unchanged.
//!
//! Run with `GLASS_BASELINE_WRITE=1` on the unpatched code to write
//! `target/glass-baseline/<name>-<aa>.png` (relative to this crate's directory).

use std::path::Path;

use vello::AaConfig;
use vello::peniko::color::palette;
use vello_research_tests::glass::{BASELINE_SIZE as SIZE, baseline_path, baseline_scenes};
use vello_research_tests::{TestParams, get_scene_image, write_png_to_file};

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
#[cfg_attr(skip_gpu_tests, ignore)]
fn stock_path_matches_the_upstream_baseline() {
    let write = std::env::var_os("GLASS_BASELINE_WRITE").is_some_and(|v| v == "1");
    if write {
        std::fs::create_dir_all("target/glass-baseline").unwrap();
    } else if !Path::new("target/glass-baseline").is_dir() {
        eprintln!(
            "glass.baseline.skipped: target/glass-baseline is absent; regenerate it from \
             f3000c8 with GLASS_BASELINE_WRITE=1"
        );
        return;
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
