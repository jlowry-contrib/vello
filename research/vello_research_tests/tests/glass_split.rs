// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glass phase 0: backdrop snapshots, the external pass hook and skipped markers.

use std::sync::{Mutex, Once};

use vello::glass::{GlassMarker, GlassResumeFormat, IRect, MarkerPlacement};
use vello::kurbo::{Affine, Circle, Rect, Shape};
use vello::peniko::{Color, Fill, Mix, color::palette};
use vello::{AaConfig, Scene};
use vello_research_tests::glass::{
    Atlas, CaptureBackdrop, background, max_delta, render_pair, render_scene,
};
use vello_research_tests::{TestParams, get_scene_image};

/// How much of a test scene to build.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Part {
    /// Everything, without glass markers.
    Unmarked,
    /// Everything, with glass markers.
    Marked,
    /// Everything up to the first marker, then the open layers closed.
    Prefix,
}

impl Part {
    fn of(marked: bool) -> Self {
        if marked { Self::Marked } else { Self::Unmarked }
    }
}

fn rect(x0: i32, y0: i32, x1: i32, y1: i32) -> IRect {
    IRect { x0, y0, x1, y1 }
}

fn marker(scene: &mut Scene, region: IRect, composite: IRect) {
    scene.push_glass(GlassMarker {
        id: 0,
        region,
        composite,
    });
}

fn fill(scene: &mut Scene, rgba: [u8; 4], shape: &impl Shape) {
    let [r, g, b, a] = rgba;
    let color = Color::from_rgba8(r, g, b, a);
    scene.fill(Fill::NonZero, Affine::IDENTITY, color, None, shape);
}

/// Pops `levels` layers.
///
/// Prefix scenes close their layers themselves: the resolver appends an `END_CLIP` for each
/// open layer but leaves them out of the layout's counts, so the GPU stages never run them.
fn close(scene: &mut Scene, levels: usize) {
    for _ in 0..levels {
        scene.pop_layer();
    }
}

fn layer(scene: &mut Scene, alpha: f32, clip: &impl Shape) {
    scene.push_layer(Fill::NonZero, Mix::Normal, alpha, Affine::IDENTITY, clip);
}

/// Snaps a clip shape's bounds outward to tiles, as binning clips the glass draw to them.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the test shapes' bounds are small pixel coordinates"
)]
fn clip_tiles(shape: &impl Shape) -> IRect {
    let b = shape.bounding_box();
    rect(
        b.x0.floor() as i32,
        b.y0.floor() as i32,
        b.x1.ceil() as i32,
        b.y1.ceil() as i32,
    )
    .snap_out(16)
}

fn intersect(a: IRect, b: IRect) -> IRect {
    rect(
        a.x0.max(b.x0),
        a.y0.max(b.y0),
        a.x1.min(b.x1),
        a.y1.min(b.y1),
    )
}

/// Asserts the marker's slot holds the prefix render inside `valid` and zero elsewhere.
fn assert_slot(
    atlas: &Atlas,
    placement: &MarkerPlacement,
    prefix: &[u8],
    width: u32,
    valid: IRect,
) {
    let region = placement.region;
    let slot = region.snap_out(16);
    let (ox, oy) = placement.backdrop_origin;
    let mut mismatches = 0;
    let mut first = None;
    let mut worst = 0_u8;
    for y in slot.y0..slot.y1 {
        for x in slot.x0..slot.x1 {
            let ax = ox + (x - slot.x0) as u32;
            let ay = oy + (y - slot.y0) as u32;
            let got = atlas.pixel(ax, ay);
            let inside = x >= valid.x0 && x < valid.x1 && y >= valid.y0 && y < valid.y1;
            let inside = inside && x < region.x1 && y < region.y1;
            let want = if inside {
                let i = ((y as u32 * width + x as u32) * 4) as usize;
                prefix[i..i + 4].try_into().unwrap()
            } else {
                [0; 4]
            };
            if got != want {
                mismatches += 1;
                worst = worst.max(max_delta(&got, &want));
                first.get_or_insert((x, y, got, want));
            }
        }
    }
    assert!(
        mismatches == 0,
        "marker {} slot: {mismatches} pixels differ, max delta {worst}, first (x, y, got, want) {:?}",
        placement.id,
        first.unwrap()
    );
}

/// Renders the marked scene split and unsplit, checks the outputs agree and returns the capture.
fn split(width: u32, aa: AaConfig, build: impl Fn(&mut Scene, Part)) -> CaptureBackdrop {
    let mut capture = CaptureBackdrop::default();
    let (unsplit, split) = render_pair(
        width,
        width,
        aa,
        GlassResumeFormat::F32,
        |scene, marked| build(scene, Part::of(marked)),
        &mut capture,
    );
    assert_eq!(max_delta(&unsplit, &split), 0, "split output differs");
    capture
}

fn prefix(width: u32, aa: AaConfig, build: impl Fn(&mut Scene, Part)) -> Vec<u8> {
    let mut scene = Scene::new();
    build(&mut scene, Part::Prefix);
    render_scene(width, width, aa, &scene)
}

/// Checks every captured slot against the prefix render, inside `valid`.
fn check_snapshot(width: u32, aa: AaConfig, valid: IRect, build: impl Fn(&mut Scene, Part) + Copy) {
    let capture = split(width, aa, build);
    assert_eq!(capture.phases, [1], "one phase");
    assert_eq!(capture.placements.len(), 1, "one marker");
    let atlas = capture.backdrop();
    let prefix = prefix(width, aa, build);
    let placement = &capture.placements[0];
    assert_slot(
        &atlas,
        placement,
        &prefix,
        width,
        intersect(placement.region, valid),
    );
}

const REGION: IRect = IRect {
    x0: 96,
    y0: 96,
    x1: 320,
    y1: 224,
};
const COMPOSITE: IRect = IRect {
    x0: 128,
    y0: 128,
    x1: 288,
    y1: 192,
};
const EVERYWHERE: IRect = IRect {
    x0: i32::MIN,
    y0: i32::MIN,
    x1: i32::MAX,
    y1: i32::MAX,
};

fn after_marker(scene: &mut Scene) {
    fill(
        scene,
        [230, 30, 40, 255],
        &Circle::new((200.0, 160.0), 60.0),
    );
}

fn plain(scene: &mut Scene, part: Part) {
    background(scene);
    if part == Part::Prefix {
        return;
    }
    if part == Part::Marked {
        marker(scene, REGION, COMPOSITE);
    }
    after_marker(scene);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn glass_frame_calls_the_hook_once_per_phase() {
    let capture = split(512, AaConfig::Area, plain);
    assert_eq!(capture.phases, [1]);
    assert_eq!(capture.frame_phases, 1);
    let expected = MarkerPlacement {
        id: 0,
        phase: 1,
        region: REGION,
        composite: COMPOSITE,
        backdrop_origin: (0, 0),
        glass_origin: (0, 0),
    };
    assert_eq!(capture.frame, [expected]);
    assert_eq!(capture.placements, [expected]);
    // The stock path ignores markers.
    let render = |part: Part| {
        let mut scene = Scene::new();
        plain(&mut scene, part);
        render_scene(512, 512, AaConfig::Area, &scene)
    };
    assert_eq!(max_delta(&render(Part::Unmarked), &render(Part::Marked)), 0);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_matches_the_prefix_render() {
    for aa in [AaConfig::Area, AaConfig::Msaa16] {
        check_snapshot(512, aa, EVERYWHERE, plain);
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn neighbouring_slots_stay_separate_at_an_unaligned_edge() {
    // Both regions snap to 112 px tall slots, so they share a shelf; the first ends past the
    // 500 px edge, so its last 12 slot columns and rows are never written.
    let right = rect(404, 404, 500, 500);
    let left = rect(0, 404, 96, 500);
    let build = |scene: &mut Scene, part: Part| {
        background(scene);
        if part == Part::Prefix {
            return;
        }
        if part == Part::Marked {
            marker(scene, right, rect(416, 416, 480, 480));
            marker(scene, left, rect(16, 416, 80, 480));
        }
        after_marker(scene);
        fill(
            scene,
            [20, 200, 90, 255],
            &Circle::new((450.0, 450.0), 40.0),
        );
    };
    let capture = split(500, AaConfig::Area, build);
    assert_eq!(capture.phases, [1]);
    let [a, b] = capture.placements[..] else {
        panic!("expected two placements, got {:?}", capture.placements);
    };
    assert_eq!(a.region, rect(400, 400, 500, 500));
    assert_eq!(b.region, rect(0, 400, 96, 500));
    assert_eq!((a.backdrop_origin, b.backdrop_origin), ((0, 0), (112, 0)));
    let atlas = capture.backdrop();
    let prefix = prefix(500, AaConfig::Area, build);
    assert_slot(&atlas, &a, &prefix, 500, a.region);
    assert_slot(&atlas, &b, &prefix, 500, b.region);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_inside_a_layer_matches_the_prefix_with_layers_closed() {
    // An unaligned clip, so the clip's edge tiles carry a fill that END_CLIP must reuse.
    let clip = Rect::new(70.0, 70.0, 441.0, 441.0);
    let build = |scene: &mut Scene, part: Part| {
        background(scene);
        layer(scene, 0.5, &clip);
        fill(
            scene,
            [30, 80, 230, 255],
            &Circle::new((180.0, 180.0), 150.0),
        );
        if part == Part::Prefix {
            close(scene, 1);
            return;
        }
        if part == Part::Marked {
            marker(scene, rect(32, 96, 256, 224), rect(64, 128, 224, 192));
        }
        after_marker(scene);
        scene.pop_layer();
    };
    check_snapshot(512, AaConfig::Area, clip_tiles(&clip), build);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_inside_a_circle_clip_and_an_opacity_layer_matches() {
    let outer = Rect::new(70.0, 70.0, 441.0, 441.0);
    let circle = Circle::new((256.0, 256.0), 180.0);
    let build = |scene: &mut Scene, part: Part| {
        background(scene);
        layer(scene, 0.6, &outer);
        scene.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &circle);
        fill(
            scene,
            [250, 200, 40, 255],
            &Rect::new(40.0, 40.0, 300.0, 400.0),
        );
        if part == Part::Prefix {
            close(scene, 2);
            return;
        }
        if part == Part::Marked {
            marker(scene, REGION, COMPOSITE);
        }
        after_marker(scene);
        scene.pop_layer();
        scene.pop_layer();
    };
    let valid = intersect(clip_tiles(&outer), clip_tiles(&circle));
    check_snapshot(512, AaConfig::Area, valid, build);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_in_a_clip_zero_tile_matches_the_prefix() {
    // The region's top-left tiles lie wholly outside the circle (the corner (112, 112) is
    // 203 px from the centre), so coarse reaches the marker there in the clip-zero state.
    let circle = Circle::new((256.0, 256.0), 150.0);
    let build = |scene: &mut Scene, part: Part| {
        background(scene);
        scene.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &circle);
        fill(
            scene,
            [120, 20, 160, 255],
            &Rect::new(0.0, 0.0, 512.0, 512.0),
        );
        if part == Part::Prefix {
            close(scene, 1);
            return;
        }
        if part == Part::Marked {
            marker(scene, rect(96, 96, 224, 224), rect(128, 128, 192, 192));
        }
        after_marker(scene);
        scene.pop_layer();
    };
    check_snapshot(512, AaConfig::Area, clip_tiles(&circle), build);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_with_five_open_layers_reads_the_spill_buffer() {
    // Five open levels, one past BLEND_STACK_SPLIT (4), so the outermost is in the spill buffer.
    let build = |scene: &mut Scene, part: Part| {
        background(scene);
        for i in 0..5_u8 {
            let inset = 24.0 + 32.0 * f64::from(i);
            layer(
                scene,
                0.9,
                &Rect::new(inset, inset, 512.0 - inset, 512.0 - inset),
            );
            fill(
                scene,
                [40 * i, 90, 200 - 30 * i, 160],
                &Rect::new(inset, inset, 256.0, 512.0 - inset),
            );
        }
        if part == Part::Prefix {
            close(scene, 5);
            return;
        }
        if part == Part::Marked {
            marker(scene, rect(160, 160, 352, 352), rect(192, 192, 320, 320));
        }
        after_marker(scene);
        for _ in 0..5 {
            scene.pop_layer();
        }
    };
    check_snapshot(512, AaConfig::Area, EVERYWHERE, build);
}

/// Collects log messages, so a test can check a warning was emitted.
struct LogCapture(Mutex<Vec<String>>);

impl log::Log for LogCapture {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            self.0.lock().unwrap().push(record.args().to_string());
        }
    }

    fn flush(&self) {}
}

static LOGS: LogCapture = LogCapture(Mutex::new(Vec::new()));

fn capture_logs() -> &'static LogCapture {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        log::set_logger(&LOGS).unwrap();
        log::set_max_level(log::LevelFilter::Warn);
    });
    &LOGS
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn marker_nested_deeper_than_eight_by_append_is_skipped() {
    let logs = capture_logs();
    let deep_id = 9009;
    let build = |scene: &mut Scene, part: Part| {
        background(scene);
        let open = |scene: &mut Scene, levels: u8, first: u8| {
            for i in first..first + levels {
                let inset = 8.0 + 16.0 * f64::from(i);
                layer(
                    scene,
                    0.9,
                    &Rect::new(inset, inset, 512.0 - inset, 512.0 - inset),
                );
                fill(
                    scene,
                    [25 * i, 120, 220 - 20 * i, 140],
                    &Rect::new(inset, inset, 300.0, 512.0 - inset),
                );
            }
        };
        open(scene, 5, 0);
        let mut inner = Scene::new();
        open(&mut inner, 4, 5);
        if part == Part::Marked {
            inner.push_glass(GlassMarker {
                id: deep_id,
                region: rect(192, 192, 320, 320),
                composite: rect(224, 224, 288, 288),
            });
        }
        fill(
            &mut inner,
            [10, 10, 10, 255],
            &Circle::new((256.0, 256.0), 30.0),
        );
        for _ in 0..4 {
            inner.pop_layer();
        }
        scene.append(&inner, None);
        for _ in 0..5 {
            scene.pop_layer();
        }
        // A shallow marker, so the frame takes the glass path and coarse meets the deep one.
        if part == Part::Marked {
            scene.push_glass(GlassMarker {
                id: 1,
                region: rect(0, 0, 64, 64),
                composite: rect(16, 16, 48, 48),
            });
        }
        fill(scene, [200, 40, 40, 255], &Rect::new(0.0, 0.0, 40.0, 40.0));
    };
    let capture = split(512, AaConfig::Area, build);
    assert_eq!(capture.phases, [1]);
    let ids: Vec<_> = capture.frame.iter().map(|p| p.id).collect();
    assert_eq!(ids, [1]);
    let warned = format!("glass.layers.too_deep id={deep_id} depth=9 max=8");
    let messages = logs.0.lock().unwrap();
    assert!(
        messages.contains(&warned),
        "missing `{warned}` in {messages:?}"
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn cpu_stages_render_markers_without_glass() {
    let render = |marked: bool| {
        let mut scene = Scene::new();
        plain(&mut scene, Part::of(marked));
        let params = TestParams {
            base_color: Some(palette::css::WHITE),
            use_cpu: true,
            ..TestParams::new("glass_cpu", 512, 512)
        };
        pollster::block_on(get_scene_image(&params, &scene)).unwrap()
    };
    let (unmarked, marked) = (render(false), render(true));
    assert!(
        unmarked.data.data() == marked.data.data(),
        "the CPU stages render markers as nothing"
    );
}
