use std::{
    f32::consts::{PI, TAU},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use crate::{
    body::Body,
    metrics::{duration_ms, MetricsSnapshot},
    quadtree::{Node, Quadtree},
};

use quarkstrom::{egui, winit::event::VirtualKeyCode, winit_input_helper::WinitInputHelper};

use palette::{rgb::Rgba, Hsluv, IntoColor};
use ultraviolet::Vec2;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

pub static PAUSED: Lazy<AtomicBool> = Lazy::new(|| false.into());
pub static UPDATE_LOCK: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

pub static FRAME_DELAY_MS: Lazy<AtomicU64> = Lazy::new(|| 0.into());
pub static USE_RANDOM_COLORS: Lazy<AtomicBool> = Lazy::new(|| false.into());
pub static COLOR_SEED: Lazy<AtomicU64> = Lazy::new(|| 0.into());
pub static NEXT_BODY_ID: Lazy<AtomicU64> = Lazy::new(|| 0.into());
pub static FUSE_ENABLED: Lazy<AtomicBool> = Lazy::new(|| true.into());
pub static FUSE_AFTER_FRAMES: Lazy<AtomicU64> = Lazy::new(|| 20.into());
pub static RESET_REQUESTED: Lazy<AtomicBool> = Lazy::new(|| false.into());

pub static BODIES: Lazy<Mutex<Vec<Body>>> = Lazy::new(|| Mutex::new(Vec::new()));
pub static QUADTREE: Lazy<Mutex<Vec<Node>>> = Lazy::new(|| Mutex::new(Vec::new()));
pub static METRICS: Lazy<Mutex<MetricsSnapshot>> = Lazy::new(|| Mutex::new(MetricsSnapshot::default()));
pub static WANT_QUADTREE: Lazy<AtomicBool> = Lazy::new(|| false.into());

pub static SPAWN: Lazy<Mutex<Vec<Body>>> = Lazy::new(|| Mutex::new(Vec::new()));

pub struct Renderer {
    pos: Vec2,
    scale: f32,

    settings_window_open: bool,

    show_bodies: bool,
    use_random_colors: bool,
    show_quadtree: bool,

    depth_range: (usize, usize),
    frame_delay_ms: u64,
    fuse_enabled: bool,
    fuse_after_frames: u64,

    spawn_body: Option<Body>,
    angle: Option<f32>,
    total: Option<f32>,

    confirmed_bodies: Option<Body>,

    bodies: Vec<Body>,
    quadtree: Vec<Node>,
    metrics: MetricsSnapshot,
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

fn color_for_id(id: u64, seed: u64) -> [u8; 4] {
    let x = mix64(id ^ seed);
    let r = ((x >> 0) & 0xff) as u8;
    let g = ((x >> 8) & 0xff) as u8;
    let b = ((x >> 16) & 0xff) as u8;
    let r = r / 2 + 96;
    let g = g / 2 + 96;
    let b = b / 2 + 96;
    [r, g, b, 0xff]
}

impl quarkstrom::Renderer for Renderer {
    fn new() -> Self {
        Self {
            pos: Vec2::zero(),
            scale: 3600.0,

            settings_window_open: false,

            show_bodies: true,
            use_random_colors: USE_RANDOM_COLORS.load(Ordering::Relaxed),
            show_quadtree: false,

            depth_range: (0, 0),
            frame_delay_ms: FRAME_DELAY_MS.load(Ordering::Relaxed),
            fuse_enabled: FUSE_ENABLED.load(Ordering::Relaxed),
            fuse_after_frames: FUSE_AFTER_FRAMES.load(Ordering::Relaxed),

            spawn_body: None,
            angle: None,
            total: None,

            confirmed_bodies: None,

            bodies: Vec::new(),
            quadtree: Vec::new(),
            metrics: MetricsSnapshot::default(),
        }
    }

    fn input(&mut self, input: &WinitInputHelper, width: u16, height: u16) {
        self.settings_window_open ^= input.key_pressed(VirtualKeyCode::E);

        if input.key_pressed(VirtualKeyCode::Space) {
            let val = PAUSED.load(Ordering::Relaxed);
            PAUSED.store(!val, Ordering::Relaxed)
        }

        if let Some((mx, my)) = input.mouse() {
            // Scroll steps to double/halve the scale
            let steps = 5.0;

            // Modify input
            let zoom = (-input.scroll_diff() / steps).exp2();

            // Screen space -> view space
            let target =
                Vec2::new(mx * 2.0 - width as f32, height as f32 - my * 2.0) / height as f32;

            // Move view position based on target
            self.pos += target * self.scale * (1.0 - zoom);

            // Zoom
            self.scale *= zoom;
        }

        // Grab
        if input.mouse_held(2) {
            let (mdx, mdy) = input.mouse_diff();
            self.pos.x -= mdx / height as f32 * self.scale * 2.0;
            self.pos.y += mdy / height as f32 * self.scale * 2.0;
        }

        let world_mouse = || -> Vec2 {
            let (mx, my) = input.mouse().unwrap_or_default();
            let mut mouse = Vec2::new(mx, my);
            mouse *= 2.0 / height as f32;
            mouse.y -= 1.0;
            mouse.y *= -1.0;
            mouse.x -= width as f32 / height as f32;
            mouse * self.scale + self.pos
        };

        if input.mouse_pressed(1) {
            let mouse = world_mouse();
            let id = NEXT_BODY_ID.fetch_add(1, Ordering::Relaxed);
            self.spawn_body = Some(Body::new(id, mouse, Vec2::zero(), 1.0, 1.0));
            self.angle = None;
            self.total = Some(0.0);
        } else if input.mouse_held(1) {
            if let Some(body) = &mut self.spawn_body {
                let mouse = world_mouse();
                if let Some(angle) = self.angle {
                    let d = mouse - body.pos;
                    let angle2 = d.y.atan2(d.x);
                    let a = angle2 - angle;
                    let a = (a + PI).rem_euclid(TAU) - PI;
                    let total = self.total.unwrap() - a;
                    body.mass = (total / TAU).exp2();
                    self.angle = Some(angle2);
                    self.total = Some(total);
                } else {
                    let d = mouse - body.pos;
                    let angle = d.y.atan2(d.x);
                    self.angle = Some(angle);
                }
                body.radius = body.mass.cbrt();
                body.vel = mouse - body.pos;
            }
        } else if input.mouse_released(1) {
            self.confirmed_bodies = self.spawn_body.take();
        }
    }

    fn render(&mut self, ctx: &mut quarkstrom::RenderContext) {
        {
            let mut lock = UPDATE_LOCK.lock();
            if *lock {
                std::mem::swap(&mut self.bodies, &mut BODIES.lock());
                std::mem::swap(&mut self.quadtree, &mut QUADTREE.lock());
                self.metrics = *METRICS.lock();
            }
            if let Some(body) = self.confirmed_bodies.take() {
                self.bodies.push(body);
                SPAWN.lock().push(body);
            }
            *lock = false;
        }

        ctx.clear_circles();
        ctx.clear_lines();
        ctx.clear_rects();
        ctx.set_view_pos(self.pos);
        ctx.set_view_scale(self.scale);

        if !self.bodies.is_empty() {
            if self.show_bodies {
                let seed = COLOR_SEED.load(Ordering::Relaxed);
                for i in 0..self.bodies.len() {
                    let color = if self.use_random_colors {
                        color_for_id(self.bodies[i].id, seed)
                    } else {
                        [0xff; 4]
                    };
                    ctx.draw_circle(self.bodies[i].pos, self.bodies[i].radius, color);
                }
            }

            if let Some(body) = &self.confirmed_bodies {
                let seed = COLOR_SEED.load(Ordering::Relaxed);
                let color = if self.use_random_colors {
                    color_for_id(body.id, seed)
                } else {
                    [0xff; 4]
                };
                ctx.draw_circle(body.pos, body.radius, color);
                ctx.draw_line(body.pos, body.pos + body.vel, [0xff; 4]);
            }

            if let Some(body) = &self.spawn_body {
                let seed = COLOR_SEED.load(Ordering::Relaxed);
                let color = if self.use_random_colors {
                    color_for_id(body.id, seed)
                } else {
                    [0xff; 4]
                };
                ctx.draw_circle(body.pos, body.radius, color);
                ctx.draw_line(body.pos, body.pos + body.vel, [0xff; 4]);
            }
        }

        if self.show_quadtree && !self.quadtree.is_empty() {
            let mut depth_range = self.depth_range;
            if depth_range.0 >= depth_range.1 {
                let mut stack = Vec::new();
                stack.push((Quadtree::ROOT, 0));

                let mut min_depth = usize::MAX;
                let mut max_depth = 0;
                while let Some((node, depth)) = stack.pop() {
                    let node = &self.quadtree[node];

                    if node.is_leaf() {
                        if depth < min_depth {
                            min_depth = depth;
                        }
                        if depth > max_depth {
                            max_depth = depth;
                        }
                    } else {
                        for i in 0..4 {
                            stack.push((node.children + i, depth + 1));
                        }
                    }
                }

                depth_range = (min_depth, max_depth);
            }
            let (min_depth, max_depth) = depth_range;

            let mut stack = Vec::new();
            stack.push((Quadtree::ROOT, 0));
            while let Some((node, depth)) = stack.pop() {
                let node = &self.quadtree[node];

                if node.is_branch() && depth < max_depth {
                    for i in 0..4 {
                        stack.push((node.children + i, depth + 1));
                    }
                } else if depth >= min_depth {
                    let quad = node.quad;
                    let half = Vec2::new(0.5, 0.5) * quad.size;
                    let min = quad.center - half;
                    let max = quad.center + half;

                    let t = ((depth - min_depth + !node.is_empty() as usize) as f32)
                        / (max_depth - min_depth + 1) as f32;

                    let start_h = -100.0;
                    let end_h = 80.0;
                    let h = start_h + (end_h - start_h) * t;
                    let s = 100.0;
                    let l = t * 100.0;

                    let c = Hsluv::new(h, s, l);
                    let rgba: Rgba = c.into_color();
                    let color = rgba.into_format().into();

                    ctx.draw_rect(min, max, color);
                }
            }
        }
    }

    fn gui(&mut self, ctx: &quarkstrom::egui::Context) {
        egui::Window::new("")
            .open(&mut self.settings_window_open)
            .show(ctx, |ui| {
                ui.checkbox(&mut self.show_bodies, "Show Bodies");
                let mut use_random_colors = self.use_random_colors;
                if ui.checkbox(&mut use_random_colors, "Random Colors").changed() {
                    self.use_random_colors = use_random_colors;
                    USE_RANDOM_COLORS.store(use_random_colors, Ordering::Relaxed);
                }

                ui.horizontal(|ui| {
                    let mut seed = COLOR_SEED.load(Ordering::Relaxed);
                    if ui.button("Re-roll Colors").clicked() {
                        seed = fastrand::u64(1..u64::MAX);
                        COLOR_SEED.store(seed, Ordering::Relaxed);
                    }
                    ui.label(format!("seed {}", seed));
                });

                if ui.checkbox(&mut self.show_quadtree, "Show Quadtree").changed() {
                    WANT_QUADTREE.store(self.show_quadtree, Ordering::Relaxed);
                }
                if self.show_quadtree {
                    let range = &mut self.depth_range;
                    ui.horizontal(|ui| {
                        ui.label("Depth Range:");
                        ui.add(egui::DragValue::new(&mut range.0).speed(0.05));
                        ui.label("to");
                        ui.add(egui::DragValue::new(&mut range.1).speed(0.05));
                    });
                }

                ui.separator();
                ui.label("Playback");
                if ui.button("Reset Simulation").clicked() {
                    RESET_REQUESTED.store(true, Ordering::Relaxed);
                }
                ui.add(
                    egui::Slider::new(&mut self.frame_delay_ms, 0..=500)
                        .text("Frame Delay (ms)"),
                );
                FRAME_DELAY_MS.store(self.frame_delay_ms, Ordering::Relaxed);

                ui.separator();
                ui.label("Fusion");
                if ui
                    .checkbox(&mut self.fuse_enabled, "Fuse Persistent Contacts")
                    .changed()
                {
                    FUSE_ENABLED.store(self.fuse_enabled, Ordering::Relaxed);
                }
                ui.add(
                    egui::Slider::new(&mut self.fuse_after_frames, 1..=240)
                        .text("Fuse After (frames)"),
                );
                FUSE_AFTER_FRAMES.store(self.fuse_after_frames, Ordering::Relaxed);

                ui.separator();
                ui.label("Timing (ms)");
                ui.label(format!("frame {}", self.metrics.frame));
                let last = self.metrics.last;
                let avg = self.metrics.avg;
                let fps = 1000.0 / duration_ms(avg.total()).max(1e-6);
                ui.label(format!("avg {:.1} fps", fps));
                egui::Grid::new("timings_grid").striped(true).show(ui, |ui| {
                    ui.label("");
                    ui.label("last");
                    ui.label("avg");
                    ui.end_row();

                    ui.label("total");
                    ui.label(format!("{:.3}", duration_ms(last.total())));
                    ui.label(format!("{:.3}", duration_ms(avg.total())));
                    ui.end_row();

                    ui.label("iterate");
                    ui.label(format!("{:.3}", duration_ms(last.iterate)));
                    ui.label(format!("{:.3}", duration_ms(avg.iterate)));
                    ui.end_row();

                    ui.label("collide");
                    ui.label(format!("{:.3}", duration_ms(last.collide)));
                    ui.label(format!("{:.3}", duration_ms(avg.collide)));
                    ui.end_row();

                    ui.label("attract total");
                    ui.label(format!("{:.3}", duration_ms(last.attract_total)));
                    ui.label(format!("{:.3}", duration_ms(avg.attract_total)));
                    ui.end_row();

                    ui.label("  build");
                    ui.label(format!("{:.3}", duration_ms(last.quadtree_build)));
                    ui.label(format!("{:.3}", duration_ms(avg.quadtree_build)));
                    ui.end_row();

                    ui.label("  acc");
                    ui.label(format!("{:.3}", duration_ms(last.quadtree_acc)));
                    ui.label(format!("{:.3}", duration_ms(avg.quadtree_acc)));
                    ui.end_row();
                });

                ui.separator();
                ui.label("Counts (avg over window)");
                let c_last = self.metrics.counts_last;
                let c_avg = self.metrics.counts_avg;
                egui::Grid::new("counts_grid").striped(true).show(ui, |ui| {
                    ui.label("");
                    ui.label("last");
                    ui.label("avg");
                    ui.end_row();

                    ui.label("bodies");
                    ui.label(format!("{}", c_last.bodies));
                    ui.label(format!("{}", c_avg.bodies));
                    ui.end_row();

                    ui.label("active quadtree nodes");
                    ui.label(format!("{}", c_last.quadtree_nodes_active));
                    ui.label(format!("{}", c_avg.quadtree_nodes_active));
                    ui.end_row();

                    ui.label("collision pairs");
                    ui.label(format!("{}", c_last.collision_pairs));
                    ui.label(format!("{}", c_avg.collision_pairs));
                    ui.end_row();
                });
            });
    }
}
