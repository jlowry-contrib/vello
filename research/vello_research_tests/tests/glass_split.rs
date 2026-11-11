// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glass split rendering: backdrop snapshots, the hook, stops, resumes and skipped markers.

use std::sync::{Mutex, Once};

use vello::glass::{
    GlassFrame, GlassMarker, GlassPasses, GlassPhase, GlassResumeFormat, IRect, MarkerPlacement,
};
use vello::kurbo::{Affine, Circle, Rect, Shape};
use vello::peniko::{Color, Fill, Mix};
use vello::wgpu::{self, CommandEncoder, Device, Queue, util::DeviceExt};
use vello::{AaConfig, AaSupport, Renderer, Scene};
use vello_research_tests::glass::{
    Atlas, CaptureBackdrop, NoGlass, background, gpu_renderer, max_delta, read_glass_state,
    render_frame, render_pair, render_pair_cpu, render_scene, validation_error,
};

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
        |_| {},
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
fn marker_with_id(scene: &mut Scene, id: u32, region: IRect, composite: IRect) {
    scene.push_glass(GlassMarker {
        id,
        region,
        composite,
    });
}

/// Renders `build`'s scene unsplit and split with `passes`, returning both outputs.
fn outputs(
    aa: AaConfig,
    resume: GlassResumeFormat,
    configure: impl Fn(&mut Renderer),
    build: impl Fn(&mut Scene, bool),
    passes: &mut dyn GlassPasses,
) -> (Vec<u8>, Vec<u8>) {
    render_pair(512, 512, aa, resume, configure, build, passes)
}

/// Asserts the split render of `build` equals the unsplit one in every byte, with `NoGlass`.
fn assert_identical(aa: AaConfig, build: impl Fn(&mut Scene, bool)) {
    let (unsplit, split) = outputs(aa, GlassResumeFormat::F32, |_| {}, build, &mut NoGlass);
    assert_eq!(max_delta(&unsplit, &split), 0, "split output differs");
}

/// Asserts the split render is within one level of the unsplit one and logs whether exact.
fn assert_within_one_level(
    name: &str,
    aa: AaConfig,
    resume: GlassResumeFormat,
    configure: impl Fn(&mut Renderer),
    build: impl Fn(&mut Scene, bool),
) -> u8 {
    let (unsplit, split) = outputs(aa, resume, configure, build, &mut NoGlass);
    let delta = max_delta(&unsplit, &split);
    let differing = unsplit
        .chunks(4)
        .zip(split.chunks(4))
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "glass.identity {name} exact={} max_delta={delta} pixels={differing}",
        delta == 0
    );
    assert!(delta <= 1, "{name}: split differs by {delta} levels");
    delta
}

fn plain_scene(scene: &mut Scene, marked: bool) {
    plain(scene, Part::of(marked));
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn one_marker_without_content_after_is_identical() {
    assert_identical(AaConfig::Area, |scene, marked| {
        background(scene);
        if marked {
            marker(scene, REGION, COMPOSITE);
        }
    });
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn content_after_the_marker_is_within_one_level_in_f32() {
    assert_within_one_level(
        "content_after_f32",
        AaConfig::Area,
        GlassResumeFormat::F32,
        |_| {},
        plain_scene,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stops_disabled_control_matches_the_stock_pipeline() {
    // The glass permutation without stops: any difference from stock is the compiler's.
    let delta = assert_within_one_level(
        "stops_disabled_control",
        AaConfig::Area,
        GlassResumeFormat::F32,
        |renderer| renderer.set_glass_stops_disabled(true),
        plain_scene,
    );
    assert_eq!(
        delta, 0,
        "the stops-disabled glass pipeline differs from stock"
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn content_after_the_marker_is_within_one_level_in_unorm16() {
    assert_within_one_level(
        "content_after_unorm16",
        AaConfig::Area,
        GlassResumeFormat::Unorm16,
        |_| {},
        plain_scene,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn content_after_the_marker_is_within_one_level_with_msaa16() {
    assert_within_one_level(
        "content_after_msaa16",
        AaConfig::Msaa16,
        GlassResumeFormat::F32,
        |_| {},
        plain_scene,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn marker_inside_an_opacity_layer_is_identical() {
    assert_identical(AaConfig::Area, |scene, marked| {
        background(scene);
        layer(scene, 0.5, &Rect::new(0.0, 0.0, 512.0, 512.0));
        fill(
            scene,
            [30, 80, 230, 255],
            &Circle::new((180.0, 180.0), 150.0),
        );
        if marked {
            marker(scene, REGION, COMPOSITE);
        }
        scene.pop_layer();
    });
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn marker_inside_a_clip_near_its_edge_is_identical() {
    // The composite crosses the clip's unaligned left edge at x = 70.
    let clip = Rect::new(70.0, 70.0, 441.0, 441.0);
    assert_identical(AaConfig::Area, |scene, marked| {
        background(scene);
        layer(scene, 0.8, &clip);
        fill(
            scene,
            [30, 80, 230, 200],
            &Circle::new((180.0, 180.0), 150.0),
        );
        if marked {
            marker(scene, rect(32, 96, 256, 224), rect(48, 112, 224, 192));
        }
        scene.pop_layer();
    });
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn five_nested_layers_resume_through_the_spill_buffer() {
    // Five open levels, one past BLEND_STACK_SPLIT (4), so the outermost is in the spill buffer.
    assert_identical(AaConfig::Area, |scene, marked| {
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
        if marked {
            marker(scene, rect(160, 160, 352, 352), rect(192, 192, 320, 320));
        }
        for _ in 0..5 {
            scene.pop_layer();
        }
    });
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn three_phases_are_identical() {
    // Each composite meets the next region, so the markers take phases 1, 2 and 3.
    let mut capture = CaptureBackdrop::default();
    let (unsplit, split) = outputs(
        AaConfig::Area,
        GlassResumeFormat::F32,
        |_| {},
        |scene, marked| {
            background(scene);
            if marked {
                marker_with_id(scene, 0, rect(64, 64, 192, 192), rect(96, 96, 160, 160));
                marker_with_id(scene, 1, rect(128, 128, 256, 256), rect(160, 160, 224, 224));
                marker_with_id(scene, 2, rect(192, 192, 320, 320), rect(224, 224, 288, 288));
            }
        },
        &mut capture,
    );
    assert_eq!(capture.phases, [1, 2, 3]);
    let phases: Vec<_> = capture.frame.iter().map(|p| (p.id, p.phase)).collect();
    assert_eq!(phases, [(0, 1), (1, 2), (2, 3)]);
    assert_eq!(max_delta(&unsplit, &split), 0, "split output differs");
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sixteen_disjoint_markers_share_one_phase() {
    let mut capture = CaptureBackdrop::default();
    let (unsplit, split) = outputs(
        AaConfig::Area,
        GlassResumeFormat::F32,
        |_| {},
        |scene, marked| {
            background(scene);
            if !marked {
                return;
            }
            for id in 0..16 {
                let x = 128 * (id % 4);
                let y = 128 * (id / 4);
                marker_with_id(
                    scene,
                    id as u32,
                    rect(x + 16, y + 16, x + 112, y + 112),
                    rect(x + 32, y + 32, x + 96, y + 96),
                );
            }
        },
        &mut capture,
    );
    assert_eq!(capture.phases, [1]);
    assert_eq!(capture.placements.len(), 16);
    assert_eq!(max_delta(&unsplit, &split), 0, "split output differs");
}

/// Premultiplied glass the phase-one hook writes over marker 0's composite.
const GLASS: [f32; 4] = [0.25, 0.0, 0.0, 0.5];

/// Writes [`GLASS`] over placement `m`'s slot in `phase`'s glass atlas.
fn write_glass(
    device: &Device,
    encoder: &mut CommandEncoder,
    phase: &GlassPhase<'_>,
    m: &MarkerPlacement,
) {
    let width = (m.composite.x1 - m.composite.x0) as u32;
    let height = (m.composite.y1 - m.composite.y0) as u32;
    // GLASS as half floats: 0.25, 0.0, 0.0 and 0.5.
    let texel = [0x3400_u16, 0, 0, 0x3800];
    let data: Vec<u8> = (0..width * height)
        .flat_map(|_| texel.iter().flat_map(|c| c.to_le_bytes()))
        .collect();
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("glass test G"),
        contents: &data,
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    assert_eq!(
        phase.glass_atlas.format(),
        wgpu::TextureFormat::Rgba16Float,
        "the default glass atlas format"
    );
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                // Eight bytes per texel; the composite is a multiple of 32 px wide.
                bytes_per_row: Some(width * 8),
                rows_per_image: None,
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: phase.glass_atlas,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: m.glass_origin.0,
                y: m.glass_origin.1,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Writes [`GLASS`] into marker 0's glass slot in phase 1 and captures the backdrops.
#[derive(Default)]
struct WriteGlass {
    capture: CaptureBackdrop,
}

impl GlassPasses for WriteGlass {
    fn prepare(&mut self, device: &Device, queue: &Queue, frame: &GlassFrame<'_>) {
        self.capture.prepare(device, queue, frame);
    }

    fn record(&mut self, device: &Device, encoder: &mut CommandEncoder, phase: &GlassPhase<'_>) {
        if phase.index == 1 {
            let m0 = phase.markers.iter().find(|p| p.id == 0).unwrap();
            write_glass(device, encoder, phase, m0);
        }
        self.capture.record(device, encoder, phase);
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn second_phase_snapshot_includes_the_first_phase_glass() {
    let m0_composite = rect(96, 96, 224, 160);
    let m1_region = rect(160, 128, 320, 256);
    let mut passes = WriteGlass::default();
    let _ = outputs(
        AaConfig::Area,
        GlassResumeFormat::F32,
        |_| {},
        |scene, marked| {
            background(scene);
            if marked {
                marker_with_id(scene, 0, rect(64, 64, 256, 192), m0_composite);
                marker_with_id(scene, 1, m1_region, rect(192, 160, 288, 224));
            }
        },
        &mut passes,
    );
    let capture = &passes.capture;
    assert_eq!(capture.phases, [1, 2]);
    let m1 = capture.frame.iter().find(|p| p.id == 1).unwrap();
    assert_eq!(m1.phase, 2);
    let mut scene = Scene::new();
    background(&mut scene);
    let backdrop = render_scene(512, 512, AaConfig::Area, &scene);
    let atlas = capture.backdrop();
    let overlap = intersect(m0_composite, m1_region);
    assert!(!overlap.is_empty(), "the test markers overlap");
    let mut worst = 0.0_f64;
    for y in m1_region.y0..m1_region.y1 {
        for x in m1_region.x0..m1_region.x1 {
            let got = atlas.pixel(
                m1.backdrop_origin.0 + (x - m1_region.x0) as u32,
                m1.backdrop_origin.1 + (y - m1_region.y0) as u32,
            );
            let i = ((y as u32 * 512 + x as u32) * 4) as usize;
            let bg = &backdrop[i..i + 4];
            let inside = x >= overlap.x0 && x < overlap.x1 && y >= overlap.y0 && y < overlap.y1;
            for c in 0..4 {
                let b = f64::from(bg[c]) / 255.0;
                let want = if inside {
                    f64::from(GLASS[c]) + (1.0 - f64::from(GLASS[3])) * b
                } else {
                    b
                };
                let delta = (f64::from(got[c]) - want * 255.0).abs();
                worst = worst.max(delta);
                assert!(
                    delta <= 1.0,
                    "({x}, {y}) channel {c}: got {}, want {:.2} (inside overlap: {inside})",
                    got[c],
                    want * 255.0
                );
            }
        }
    }
    eprintln!("glass.second_phase max_delta={worst:.3}");
}

/// Writes [`GLASS`] into every marker's glass slot in every phase and captures the backdrops.
#[derive(Default)]
struct GlassOnEveryMarker {
    capture: CaptureBackdrop,
}

impl GlassPasses for GlassOnEveryMarker {
    fn prepare(&mut self, device: &Device, queue: &Queue, frame: &GlassFrame<'_>) {
        self.capture.prepare(device, queue, frame);
    }

    fn record(&mut self, device: &Device, encoder: &mut CommandEncoder, phase: &GlassPhase<'_>) {
        for m in phase.markers {
            write_glass(device, encoder, phase, m);
        }
        self.capture.record(device, encoder, phase);
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn marker_past_the_phase_cap_draws_nothing() {
    // Each composite meets the next region, so the fifth marker would need phase 5.
    let chain = |count: i32| {
        move |scene: &mut Scene, marked: bool| {
            background(scene);
            if marked {
                for i in 0..count {
                    let o = 64 * i;
                    let region = rect(64 + o, 64 + o, 192 + o, 192 + o);
                    let composite = rect(96 + o, 96 + o, 160 + o, 160 + o);
                    marker_with_id(scene, i as u32, region, composite);
                }
            }
        }
    };
    let render = |count: i32| {
        let mut passes = GlassOnEveryMarker::default();
        let (_, split) = outputs(
            AaConfig::Area,
            GlassResumeFormat::F32,
            |_| {},
            chain(count),
            &mut passes,
        );
        (split, passes.capture)
    };
    let (four, four_capture) = render(4);
    let (five, five_capture) = render(5);
    let phases = |c: &CaptureBackdrop| c.frame.iter().map(|p| (p.id, p.phase)).collect::<Vec<_>>();
    assert_eq!(phases(&four_capture), [(0, 1), (1, 2), (2, 3), (3, 4)]);
    assert_eq!(phases(&five_capture), phases(&four_capture));
    assert_eq!(five_capture.frame, four_capture.frame);
    assert_eq!(five_capture.phases, [1, 2, 3, 4]);
    assert_eq!(
        max_delta(&four, &five),
        0,
        "the capped marker changed the output"
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn marker_partly_off_screen_is_identical() {
    let mut capture = CaptureBackdrop::default();
    let (unsplit, split) = outputs(
        AaConfig::Area,
        GlassResumeFormat::F32,
        |_| {},
        |scene, marked| {
            background(scene);
            if marked {
                marker(scene, rect(-64, 400, 200, 600), rect(-32, 432, 160, 560));
            }
        },
        &mut capture,
    );
    assert_eq!(capture.phases, [1]);
    assert_eq!(capture.frame[0].region, rect(0, 400, 208, 512));
    assert_eq!(capture.frame[0].composite, rect(0, 432, 160, 512));
    assert_eq!(max_delta(&unsplit, &split), 0, "split output differs");
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn marker_wholly_off_screen_is_skipped() {
    let mut capture = CaptureBackdrop::default();
    let (unsplit, split) = outputs(
        AaConfig::Area,
        GlassResumeFormat::F32,
        |_| {},
        |scene, marked| {
            background(scene);
            if marked {
                marker(scene, rect(600, 600, 700, 700), rect(616, 616, 684, 684));
            }
            after_marker(scene);
        },
        &mut capture,
    );
    assert!(capture.frame.is_empty(), "prepare was not called");
    assert!(capture.phases.is_empty(), "record was not called");
    assert_eq!(max_delta(&unsplit, &split), 0, "split output differs");
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn render_to_texture_ignores_markers() {
    let clip = Rect::new(70.0, 70.0, 441.0, 441.0);
    let render = |marked: bool| {
        let mut scene = Scene::new();
        background(&mut scene);
        layer(&mut scene, 0.7, &clip);
        if marked {
            marker(&mut scene, REGION, COMPOSITE);
        }
        after_marker(&mut scene);
        scene.pop_layer();
        if marked {
            marker(&mut scene, rect(0, 0, 64, 64), rect(16, 16, 48, 48));
        }
        render_scene(512, 512, AaConfig::Area, &scene)
    };
    assert_eq!(
        max_delta(&render(false), &render(true)),
        0,
        "markers changed the stock path"
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn cpu_stages_ignore_glass_markers() {
    let logs = capture_logs();
    let mut capture = CaptureBackdrop::default();
    let (unmarked, marked) = render_pair_cpu(512, 512, plain_scene, &mut capture);
    assert_eq!(
        max_delta(&unmarked, &marked),
        0,
        "the CPU stages render markers as nothing"
    );
    assert!(capture.phases.is_empty(), "the hook is not called");
    let messages = logs.0.lock().unwrap();
    let warned = "glass.cpu.unsupported markers=1";
    assert!(
        messages.iter().any(|m| m == warned),
        "missing `{warned}` in {messages:?}"
    );
}

fn marked_scene() -> Scene {
    let mut scene = Scene::new();
    background(&mut scene);
    marker(&mut scene, REGION, COMPOSITE);
    after_marker(&mut scene);
    scene
}

fn unmarked_scene() -> Scene {
    let mut scene = Scene::new();
    background(&mut scene);
    after_marker(&mut scene);
    scene
}

fn frame(renderer: &mut Renderer, scene: &Scene, glass: bool) -> Vec<u8> {
    let passes: Option<&mut dyn GlassPasses> = if glass { Some(&mut NoGlass) } else { None };
    render_frame(renderer, 512, 512, AaConfig::Area, scene, passes)
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn renderer_new_compiles_no_glass_pipeline() {
    let renderer = gpu_renderer(AaSupport::all());
    assert!(!renderer.glass_pipelines_compiled());
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stock_frames_compile_no_glass_pipeline() {
    let mut renderer = gpu_renderer(AaSupport::all());
    let scene = unmarked_scene();
    for _ in 0..3 {
        frame(&mut renderer, &scene, false);
    }
    frame(&mut renderer, &scene, true);
    assert!(!renderer.glass_pipelines_compiled());
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn first_glass_frame_compiles_glass_pipelines_once() {
    let mut renderer = gpu_renderer(AaSupport::all());
    let scene = marked_scene();
    frame(&mut renderer, &scene, true);
    assert!(renderer.glass_pipelines_compiled());
    let after_first = renderer.shader_count();
    frame(&mut renderer, &scene, true);
    assert_eq!(
        renderer.shader_count(),
        after_first,
        "second frame added shaders"
    );
}

#[cfg(feature = "hot_reload")]
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn reload_shaders_resets_glass_pipelines() {
    let mut renderer = gpu_renderer(AaSupport::all());
    let scene = marked_scene();
    let before = frame(&mut renderer, &scene, true);
    vello_research_tests::glass::reload_shaders(&mut renderer);
    assert!(!renderer.glass_pipelines_compiled());
    let after = frame(&mut renderer, &scene, true);
    assert!(renderer.glass_pipelines_compiled());
    assert_eq!(
        max_delta(&before, &after),
        0,
        "output changed across the reload"
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn no_marker_scene_is_identical_through_both_entry_points() {
    let mut renderer = gpu_renderer(AaSupport::all());
    let scene = unmarked_scene();
    let stock = frame(&mut renderer, &scene, false);
    let glass = frame(&mut renderer, &scene, true);
    assert_eq!(max_delta(&stock, &glass), 0, "entry points differ");
}

/// Sixteen disjoint markers, each composite stopping its 16 tiles for phase 1.
fn sixteen_markers() -> Scene {
    let mut scene = Scene::new();
    background(&mut scene);
    for id in 0..16 {
        let x = 128 * (id % 4);
        let y = 128 * (id / 4);
        marker_with_id(
            &mut scene,
            id as u32,
            rect(x + 16, y + 16, x + 112, y + 112),
            rect(x + 32, y + 32, x + 96, y + 96),
        );
    }
    scene
}

/// Returns `glass_state` as `u32` words.
fn glass_state_words(renderer: &Renderer) -> Vec<u32> {
    let bytes = read_glass_state(renderer);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&w| u32::from_ne_bytes(w))
        .collect()
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn glass_state_clear_covers_only_the_header() {
    // `glass_state`'s header in words: the running phase, list 0's length, two free words,
    // four stop counts.
    const HEADER_WORDS: usize = 8;
    let mut renderer = gpu_renderer(AaSupport::area_only());
    let scene = sixteen_markers();
    frame(&mut renderer, &scene, true);
    let first = glass_state_words(&renderer);
    // List 0 holds 576 tiles, 16 regions of 6 × 6, and phase 1 ran once and stopped 256
    // tiles, 16 composites of 4 × 4.
    assert_eq!(first[..HEADER_WORDS], [1, 576, 0, 0, 256, 0, 0, 0]);
    assert!(
        first[HEADER_WORDS..].iter().any(|&w| w != 0),
        "the first frame wrote no resume records"
    );
    // The same scene sizes `glass_state` alike, so the second frame reuses the pooled buffer.
    renderer.set_glass_stops_disabled(true);
    frame(&mut renderer, &scene, true);
    let second = glass_state_words(&renderer);
    assert_eq!(second.len(), first.len(), "the pooled buffer changed");
    assert_eq!(second[..HEADER_WORDS], [1, 576, 0, 0, 0, 0, 0, 0]);
    // List 0 is rewritten each frame, in whatever order the tiles ran; the phase lists and
    // resume records after it are written only by stops.
    let records = HEADER_WORDS + 576;
    assert!(
        second[records..] == first[records..],
        "the clear reached past the header"
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn forced_glass_permutation_without_markers_is_identical() {
    let scene = unmarked_scene();
    for aa in [AaConfig::Area, AaConfig::Msaa16] {
        let mut renderer = gpu_renderer(AaSupport::all());
        let stock = render_frame(&mut renderer, 512, 512, aa, &scene, None);
        renderer.set_glass_permutation_forced(true);
        let mut capture = CaptureBackdrop::default();
        let forced = render_frame(&mut renderer, 512, 512, aa, &scene, Some(&mut capture));
        assert!(
            renderer.glass_pipelines_compiled(),
            "{aa:?}: the forced frame compiled no glass pipeline"
        );
        assert!(capture.frame.is_empty() && capture.frame_phases == 0);
        assert!(capture.phases.is_empty(), "{aa:?}: a phase was recorded");
        // No tile has glass commands and no phase ran, so the header stays as cleared.
        assert_eq!(glass_state_words(&renderer)[..8], [0; 8]);
        let delta = max_delta(&stock, &forced);
        // The glass permutation may round differently under fast math, by at most one level.
        eprintln!(
            "glass.identity forced_{aa:?} exact={} max_delta={delta}",
            delta == 0
        );
        assert!(
            delta <= 1,
            "{aa:?}: forcing changed the output by {delta} levels"
        );
    }
}

/// Returns the indices of the 16 px tiles of `rects`, snapped outward and clipped to a
/// `width`×`height` target, in order and without repeats.
fn tiles_of(rects: &[IRect], width: u32, height: u32) -> Vec<u32> {
    let (columns, rows) = (width.div_ceil(16), height.div_ceil(16));
    let mut tiles: Vec<u32> = rects
        .iter()
        .flat_map(|r| {
            let r = r.snap_out(16);
            let [x0, y0] = [r.x0, r.y0].map(|v| (v.max(0) / 16) as u32);
            let x1 = ((r.x1.max(0) / 16) as u32).min(columns);
            let y1 = ((r.y1.max(0) / 16) as u32).min(rows);
            (y0..y1).flat_map(move |y| (x0..x1).map(move |x| y * columns + x))
        })
        .collect();
    tiles.sort_unstable();
    tiles.dedup();
    tiles
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn lite_fine_leaves_glass_tiles_unwritten() {
    let glass_tiles = tiles_of(&[REGION], 512, 512);
    for aa in [AaConfig::Area, AaConfig::Msaa8, AaConfig::Msaa16] {
        let mut renderer = gpu_renderer(AaSupport::all());
        let mut stock_scene = Scene::new();
        plain(&mut stock_scene, Part::Unmarked);
        let stock = render_frame(&mut renderer, 512, 512, aa, &stock_scene, None);
        let mut scene = Scene::new();
        plain(&mut scene, Part::Marked);
        renderer.set_glass_start_pass_disabled(true);
        let lite = render_frame(&mut renderer, 512, 512, aa, &scene, Some(&mut NoGlass));
        let pixels = (0..512_u32).flat_map(|y| (0..512_u32).map(move |x| (x, y)));
        for ((x, y), (stock_px, lite_px)) in pixels.zip(stock.chunks(4).zip(lite.chunks(4))) {
            let tile = (y / 16) * 32 + x / 16;
            if glass_tiles.binary_search(&tile).is_ok() {
                // The target starts zeroed, and no pass wrote it.
                assert_eq!(
                    lite_px, [0; 4],
                    "{aa:?}: glass pixel ({x}, {y}) was written"
                );
            } else {
                assert_eq!(
                    lite_px, stock_px,
                    "{aa:?}: pixel ({x}, {y}) differs from stock"
                );
            }
        }
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn glass_tile_list_holds_exactly_the_tiles_with_glass_commands() {
    // The list of tiles with glass commands, after the 8-word header; its count is word 1.
    const HEADER_WORDS: usize = 8;
    let mut scene = sixteen_markers();
    // A seventeenth marker that reaches past the target's corner.
    let corner = (rect(500, 500, 600, 600), rect(510, 510, 590, 590));
    marker_with_id(&mut scene, 16, corner.0, corner.1);
    let mut regions: Vec<IRect> = (0..16)
        .map(|id| {
            let (x, y) = (128 * (id % 4), 128 * (id / 4));
            rect(x + 16, y + 16, x + 112, y + 112)
        })
        .collect();
    regions.push(corner.0);
    let expected = tiles_of(&regions, 512, 512);
    for aa in [AaConfig::Area, AaConfig::Msaa16] {
        let mut renderer = gpu_renderer(AaSupport::all());
        render_frame(&mut renderer, 512, 512, aa, &scene, Some(&mut NoGlass));
        let words = glass_state_words(&renderer);
        let count = words[1] as usize;
        let mut list = words[HEADER_WORDS..HEADER_WORDS + count].to_vec();
        list.sort_unstable();
        assert_eq!(list, expected, "{aa:?}: the glass tile list differs");
    }
}

/// Returns the lite pass's command count of `scene`'s glass frame: `glass_state` word 3.
fn lite_commands(aa: AaConfig, scene: &Scene) -> u32 {
    let mut renderer = gpu_renderer(AaSupport::all());
    renderer.set_glass_lite_commands_counted(true);
    render_frame(&mut renderer, 512, 512, aa, scene, Some(&mut NoGlass));
    glass_state_words(&renderer)[3]
}

/// The background, eight squares inside `area`, then the usual marker.
fn squares_then_marker(area: Option<Rect>) -> Scene {
    let mut scene = Scene::new();
    background(&mut scene);
    if let Some(area) = area {
        for i in 0..8 {
            let inset = 4.0 * f64::from(i);
            let square = area.inset(-inset);
            fill(&mut scene, [40, 200, 90, 128], &square);
        }
    }
    marker(&mut scene, REGION, COMPOSITE);
    scene
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn lite_pass_renders_no_command_of_a_flagged_tile() {
    // Squares inside the marker's region add commands only to flagged tiles; squares
    // outside it add them to tiles the lite pass renders.
    let inside = Rect::new(112.0, 112.0, 304.0, 208.0);
    let outside = Rect::new(352.0, 352.0, 496.0, 496.0);
    for aa in [AaConfig::Area, AaConfig::Msaa16] {
        let plain = lite_commands(aa, &squares_then_marker(None));
        let with_inside = lite_commands(aa, &squares_then_marker(Some(inside)));
        let with_outside = lite_commands(aa, &squares_then_marker(Some(outside)));
        assert!(
            with_outside > plain,
            "{aa:?}: the count missed commands outside the region"
        );
        assert_eq!(
            with_inside, plain,
            "{aa:?}: the lite pass ran commands of a flagged tile"
        );
    }
}

/// The four-marker chain of `marker_past_the_phase_cap_draws_nothing`, plus a fifth marker
/// past the phase cap when `capped`.
fn chain(capped: bool) -> Scene {
    let mut scene = Scene::new();
    background(&mut scene);
    for i in 0..if capped { 5 } else { 4 } {
        let o = 64 * i;
        let region = rect(64 + o, 64 + o, 192 + o, 192 + o);
        let composite = rect(96 + o, 96 + o, 160 + o, 160 + o);
        marker_with_id(&mut scene, i as u32, region, composite);
    }
    scene
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn capped_marker_emits_no_glass_commands() {
    let header = |capped: bool| {
        let mut renderer = gpu_renderer(AaSupport::area_only());
        frame(&mut renderer, &chain(capped), true);
        glass_state_words(&renderer)[..8].to_vec()
    };
    let (four, five) = (header(false), header(true));
    // The fifth region reaches tiles no other region covers, so any glass command coarse
    // wrote for the capped marker would add them to list 0, word 1.
    assert_eq!(
        five, four,
        "the capped marker changed the glass state header"
    );
    // Four phases ran and each stopped its 16 composite tiles; nothing overflowed.
    assert_eq!(five[0], 4);
    assert_eq!(five[2..], [0, 0, 16, 16, 16, 16]);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn resume_setup_clamps_to_capacity() {
    let scene = sixteen_markers();
    // Phase 1 stops 256 tiles into a capacity of 200.
    let mut renderer = gpu_renderer(AaSupport::area_only());
    renderer.set_glass_capacity_for_tests(None, Some(200));
    frame(&mut renderer, &scene, true);
    let words = glass_state_words(&renderer);
    assert_eq!(words[..5], [1, 576, 56, 0, 256]);
    // List 0 holds 576 tiles in a capacity of 500; with no stops, only list 0 overflows.
    let mut renderer = gpu_renderer(AaSupport::area_only());
    renderer.set_glass_stops_disabled(true);
    renderer.set_glass_capacity_for_tests(Some(500), None);
    frame(&mut renderer, &scene, true);
    let words = glass_state_words(&renderer);
    assert_eq!(words[..5], [1, 576, 76, 0, 0]);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn full_screen_composite_renders_without_a_validation_error() {
    // An iPhone 17 Pro Max screen, in pixels.
    let (width, height) = (1206, 2622);
    let full = Rect::new(0.0, 0.0, f64::from(width), f64::from(height));
    let mut scene = Scene::new();
    background(&mut scene);
    for alpha in [0.9, 0.8] {
        layer(&mut scene, alpha, &full);
    }
    let screen = rect(0, 0, width as i32, height as i32);
    marker(&mut scene, screen, screen);
    close(&mut scene, 2);
    // Depth-2 records of 1540 words make about 77 MB, which pools to 96 MiB; the old
    // 2052-word records pooled to 128 MiB and would leave the marker out under this limit.
    let limit = 96 << 20;
    let mut renderer = gpu_renderer(AaSupport::area_only());
    renderer.set_glass_state_limit_for_tests(Some(limit));
    let mut capture = CaptureBackdrop::default();
    let error = validation_error(|| {
        render_frame(
            &mut renderer,
            width,
            height,
            AaConfig::Area,
            &scene,
            Some(&mut capture),
        );
    });
    assert!(error.is_none(), "validation error: {error:?}");
    assert_eq!(capture.frame.len(), 1, "the marker was left out");
    assert_eq!(capture.phases, [1]);
    let state = read_glass_state(&renderer).len() as u64;
    assert!(state <= limit, "glass_state is {state} bytes, over {limit}");
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn marker_nested_past_eight_in_clip_zero_tiles_leaves_its_tiles_drawn() {
    // Nine layers open at the deep marker, but the ninth clips its tiles to nothing, so
    // coarse sees eight there. The frame leaves the marker out, and coarse must list none
    // of its tiles: list 0 has room for the shallow marker's only.
    // The layers cover only the deep marker's area, keeping the blend spill small.
    let area = Rect::new(192.0, 192.0, 448.0, 448.0);
    let build = |scene: &mut Scene, marked: bool| {
        background(scene);
        for _ in 0..5 {
            layer(scene, 0.9, &area);
        }
        let mut inner = Scene::new();
        for _ in 0..3 {
            layer(&mut inner, 0.9, &area);
        }
        // Two corner squares: binning keeps the deep marker, inside the clip's bounds, but
        // the clip covers none of its tiles.
        let mut corners = Rect::new(192.0, 192.0, 208.0, 208.0).to_path(0.1);
        corners.extend(Rect::new(432.0, 432.0, 448.0, 448.0).path_elements(0.1));
        layer(&mut inner, 1.0, &corners);
        if marked {
            marker_with_id(
                &mut inner,
                9,
                rect(256, 256, 384, 384),
                rect(288, 288, 352, 352),
            );
        }
        fill(
            &mut inner,
            [10, 10, 10, 255],
            &Circle::new((200.0, 200.0), 12.0),
        );
        close(&mut inner, 4);
        scene.append(&inner, None);
        close(scene, 5);
        if marked {
            marker_with_id(scene, 1, rect(448, 448, 512, 512), rect(464, 464, 496, 496));
        }
    };
    let mut renderer = gpu_renderer(AaSupport::area_only());
    let mut unmarked = Scene::new();
    build(&mut unmarked, false);
    let stock = render_frame(&mut renderer, 512, 512, AaConfig::Area, &unmarked, None);
    let mut marked = Scene::new();
    build(&mut marked, true);
    let split = render_frame(
        &mut renderer,
        512,
        512,
        AaConfig::Area,
        &marked,
        Some(&mut NoGlass),
    );
    let words = glass_state_words(&renderer);
    assert_ne!(stock[..4], [0; 4], "the stock render failed");
    assert_eq!(
        words[1..3],
        [16, 0],
        "list 0 holds tiles past the shallow marker's"
    );
    assert_eq!(max_delta(&stock, &split), 0, "split output differs");
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn state_limit_leaves_markers_out_and_logs_on_change() {
    let logs = capture_logs();
    // Each of the sixteen markers needs 36 list 0 words and 16 depth-0 `F32` records of
    // 1029 words with their list words, 66000 bytes; 1 MiB holds fifteen.
    let limit = 1 << 20;
    let mut renderer = gpu_renderer(AaSupport::area_only());
    renderer.set_glass_state_limit_for_tests(Some(limit));
    let scene = sixteen_markers();
    for _ in 0..2 {
        let mut capture = CaptureBackdrop::default();
        render_frame(
            &mut renderer,
            512,
            512,
            AaConfig::Area,
            &scene,
            Some(&mut capture),
        );
        let ids: Vec<_> = capture.frame.iter().map(|p| p.id).collect();
        assert_eq!(ids, (0..15).collect::<Vec<_>>());
    }
    let message = format!(
        "glass.state.full left_out=1 over_size=1 limit={limit} over_dispatch=0 workgroups=65535"
    );
    let count = logs
        .0
        .lock()
        .unwrap()
        .iter()
        .filter(|m| **m == message)
        .count();
    assert_eq!(
        count, 1,
        "`{message}` should be logged once, when the count changes"
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn append_under_a_fractional_translation_logs_and_draws_no_glass() {
    let logs = capture_logs();
    let mut child = Scene::new();
    marker_with_id(&mut child, 7, REGION, COMPOSITE);
    marker_with_id(
        &mut child,
        8,
        rect(400, 400, 480, 480),
        rect(416, 416, 464, 464),
    );
    let mut scene = Scene::new();
    background(&mut scene);
    scene.append(&child, Some(Affine::translate((0.5, 0.0))));
    assert!(scene.encoding().glass.is_empty());
    let message = "glass.append.non_integer markers=2".to_owned();
    assert!(
        logs.0.lock().unwrap().contains(&message),
        "missing `{message}`"
    );
    let mut unmarked = Scene::new();
    background(&mut unmarked);
    let stock = render_scene(512, 512, AaConfig::Area, &unmarked);
    let mut renderer = gpu_renderer(AaSupport::area_only());
    let glass = render_frame(
        &mut renderer,
        512,
        512,
        AaConfig::Area,
        &scene,
        Some(&mut NoGlass),
    );
    assert_eq!(
        max_delta(&stock, &glass),
        0,
        "the left-out markers changed the output"
    );
}
