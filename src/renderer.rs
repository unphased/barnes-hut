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
pub static FUSE_ENABLED: Lazy<AtomicBool> = Lazy::new(|| false.into());
pub static FUSE_AFTER_FRAMES: Lazy<AtomicU64> = Lazy::new(|| 20.into());
pub static RESET_REQUESTED: Lazy<AtomicBool> = Lazy::new(|| false.into());
pub static INIT_PARTICLES: Lazy<AtomicU64> = Lazy::new(|| 200_000.into());

pub static COLLISIONS_ENABLED: Lazy<AtomicBool> = Lazy::new(|| true.into());

pub static BONDS_ENABLED: Lazy<AtomicBool> = Lazy::new(|| false.into());
pub static BOND_AFTER_FRAMES: Lazy<AtomicU64> = Lazy::new(|| 15.into());
pub static MAX_BONDS_PER_BODY: Lazy<AtomicU64> = Lazy::new(|| 4.into());
pub static BOND_ITERS: Lazy<AtomicU64> = Lazy::new(|| 2.into());
pub static BOND_BREAK_SPEED: Lazy<AtomicU64> = Lazy::new(|| 2500.into()); // 2.5 units/s
pub static BOND_BREAK_ERROR: Lazy<AtomicU64> = Lazy::new(|| 5000.into()); // 5.0 units

pub static SPAWN_RANDOMIZE: Lazy<AtomicBool> = Lazy::new(|| false.into());
pub static INIT_RANDOMIZE: Lazy<AtomicBool> = Lazy::new(|| false.into());
pub static LINK_RADIUS_TO_MASS: Lazy<AtomicBool> = Lazy::new(|| true.into());
pub static MASS_MIN_MILLI: Lazy<AtomicU64> = Lazy::new(|| 1000.into());
pub static MASS_MAX_MILLI: Lazy<AtomicU64> = Lazy::new(|| 1000.into());
pub static DIAM_MIN_MILLI: Lazy<AtomicU64> = Lazy::new(|| 2000.into());
pub static DIAM_MAX_MILLI: Lazy<AtomicU64> = Lazy::new(|| 2000.into());

pub static VEL_FILTER_ALPHA_MILLI: Lazy<AtomicU64> = Lazy::new(|| 1000.into()); // 1.0 = off
pub static VEL_FILTER_CONTACTS_ONLY: Lazy<AtomicBool> = Lazy::new(|| true.into());
pub static VEL_FILTER_MIN_CONTACTS: Lazy<AtomicU64> = Lazy::new(|| 2.into());

pub static CLUMP_DAMP_ENABLED: Lazy<AtomicBool> = Lazy::new(|| true.into());
pub static CLUMP_DAMP_CONTACTS_ONLY: Lazy<AtomicBool> = Lazy::new(|| true.into());
pub static CLUMP_DAMP_MIN_CONTACTS: Lazy<AtomicU64> = Lazy::new(|| 4.into());
pub static CLUMP_DAMP_SPEED_MILLI: Lazy<AtomicU64> = Lazy::new(|| 800.into()); // 0.8 units/s
pub static CLUMP_DAMP_E_LOW_MILLI: Lazy<AtomicU64> = Lazy::new(|| 0.into()); // 0.0
pub static CLUMP_DAMP_E_HIGH_MILLI: Lazy<AtomicU64> = Lazy::new(|| 500.into()); // 0.5 (current)

pub static BODIES: Lazy<Mutex<Vec<Body>>> = Lazy::new(|| Mutex::new(Vec::new()));
pub static QUADTREE: Lazy<Mutex<Vec<Node>>> = Lazy::new(|| Mutex::new(Vec::new()));
pub static METRICS: Lazy<Mutex<MetricsSnapshot>> = Lazy::new(|| Mutex::new(MetricsSnapshot::default()));
pub static WANT_QUADTREE: Lazy<AtomicBool> = Lazy::new(|| false.into());
pub static WANT_BONDS: Lazy<AtomicBool> = Lazy::new(|| false.into());
pub static MAX_BOND_LINES: Lazy<AtomicU64> = Lazy::new(|| 20000.into());
pub static BOND_LINE_PX: Lazy<AtomicU64> = Lazy::new(|| 3.into());
pub static BOND_LINES: Lazy<Mutex<Vec<(Vec2, Vec2)>>> = Lazy::new(|| Mutex::new(Vec::new()));

pub static SPAWN: Lazy<Mutex<Vec<Body>>> = Lazy::new(|| Mutex::new(Vec::new()));

pub struct Renderer {
    pos: Vec2,
    scale: f32,
    viewport_height: u16,

    settings_window_open: bool,

    show_bodies: bool,
    use_random_colors: bool,
    show_quadtree: bool,
    show_bonds: bool,

    depth_range: (usize, usize),
    frame_delay_ms: u64,
    init_particles: u64,
    collisions_enabled: bool,
    fuse_enabled: bool,
    fuse_after_frames: u64,
    bonds_enabled: bool,
    bond_after_frames: u64,
    max_bonds_per_body: u64,
    bond_iters: u64,
    bond_break_speed_milli: u64,
    bond_break_error_milli: u64,
    spawn_randomize: bool,
    init_randomize: bool,
    link_radius_to_mass: bool,
    mass_min_milli: u64,
    mass_max_milli: u64,
    diam_min_milli: u64,
    diam_max_milli: u64,
    vel_filter_alpha_milli: u64,
    vel_filter_contacts_only: bool,
    vel_filter_min_contacts: u64,

    clump_damp_enabled: bool,
    clump_damp_contacts_only: bool,
    clump_damp_min_contacts: u64,
    clump_damp_speed_milli: u64,
    clump_damp_e_low_milli: u64,
    clump_damp_e_high_milli: u64,

    spawn_body: Option<Body>,
    angle: Option<f32>,
    total: Option<f32>,

    confirmed_bodies: Option<Body>,

    bodies: Vec<Body>,
    quadtree: Vec<Node>,
    bond_lines: Vec<(Vec2, Vec2)>,
    metrics: MetricsSnapshot,

    bond_line_px: u64,

    click_mode_follow: bool,
    follow_id: Option<u64>,
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
            viewport_height: 900,

            settings_window_open: false,

            show_bodies: true,
            use_random_colors: USE_RANDOM_COLORS.load(Ordering::Relaxed),
            show_quadtree: false,
            show_bonds: false,

            depth_range: (0, 0),
            frame_delay_ms: FRAME_DELAY_MS.load(Ordering::Relaxed),
            init_particles: INIT_PARTICLES.load(Ordering::Relaxed),
            collisions_enabled: COLLISIONS_ENABLED.load(Ordering::Relaxed),
            fuse_enabled: FUSE_ENABLED.load(Ordering::Relaxed),
            fuse_after_frames: FUSE_AFTER_FRAMES.load(Ordering::Relaxed),
            bonds_enabled: BONDS_ENABLED.load(Ordering::Relaxed),
            bond_after_frames: BOND_AFTER_FRAMES.load(Ordering::Relaxed),
            max_bonds_per_body: MAX_BONDS_PER_BODY.load(Ordering::Relaxed),
            bond_iters: BOND_ITERS.load(Ordering::Relaxed),
            bond_break_speed_milli: BOND_BREAK_SPEED.load(Ordering::Relaxed),
            bond_break_error_milli: BOND_BREAK_ERROR.load(Ordering::Relaxed),
            spawn_randomize: SPAWN_RANDOMIZE.load(Ordering::Relaxed),
            init_randomize: INIT_RANDOMIZE.load(Ordering::Relaxed),
            link_radius_to_mass: LINK_RADIUS_TO_MASS.load(Ordering::Relaxed),
            mass_min_milli: MASS_MIN_MILLI.load(Ordering::Relaxed),
            mass_max_milli: MASS_MAX_MILLI.load(Ordering::Relaxed),
            diam_min_milli: DIAM_MIN_MILLI.load(Ordering::Relaxed),
            diam_max_milli: DIAM_MAX_MILLI.load(Ordering::Relaxed),
            vel_filter_alpha_milli: VEL_FILTER_ALPHA_MILLI.load(Ordering::Relaxed),
            vel_filter_contacts_only: VEL_FILTER_CONTACTS_ONLY.load(Ordering::Relaxed),
            vel_filter_min_contacts: VEL_FILTER_MIN_CONTACTS.load(Ordering::Relaxed),

            clump_damp_enabled: CLUMP_DAMP_ENABLED.load(Ordering::Relaxed),
            clump_damp_contacts_only: CLUMP_DAMP_CONTACTS_ONLY.load(Ordering::Relaxed),
            clump_damp_min_contacts: CLUMP_DAMP_MIN_CONTACTS.load(Ordering::Relaxed),
            clump_damp_speed_milli: CLUMP_DAMP_SPEED_MILLI.load(Ordering::Relaxed),
            clump_damp_e_low_milli: CLUMP_DAMP_E_LOW_MILLI.load(Ordering::Relaxed),
            clump_damp_e_high_milli: CLUMP_DAMP_E_HIGH_MILLI.load(Ordering::Relaxed),

            spawn_body: None,
            angle: None,
            total: None,

            confirmed_bodies: None,

            bodies: Vec::new(),
            quadtree: Vec::new(),
            bond_lines: Vec::new(),
            metrics: MetricsSnapshot::default(),

            bond_line_px: BOND_LINE_PX.load(Ordering::Relaxed),

            click_mode_follow: false,
            follow_id: None,
        }
    }

    fn input(&mut self, input: &WinitInputHelper, width: u16, height: u16) {
        self.viewport_height = height.max(1);
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

            // Move view position based on target (unless we're following)
            if self.follow_id.is_none() {
                self.pos += target * self.scale * (1.0 - zoom);
            }

            // Zoom
            self.scale *= zoom;
        }

        // Grab
        if input.mouse_held(2) && self.follow_id.is_none() {
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

        if self.click_mode_follow && input.mouse_pressed(0) && !self.bodies.is_empty() {
            let mouse = world_mouse();
            let mut best: Option<(u64, f32)> = None;
            for body in &self.bodies {
                let d_sq = (body.pos - mouse).mag_sq();
                match best {
                    None => best = Some((body.id, d_sq)),
                    Some((_, best_sq)) if d_sq < best_sq => best = Some((body.id, d_sq)),
                    _ => {}
                }
            }

            // Require a reasonably close click (in screen pixels).
            let units_per_pixel = self.scale * 2.0 / height.max(1) as f32;
            let max_dist = units_per_pixel * 12.0;
            let max_dist_sq = max_dist * max_dist;
            if let Some((id, d_sq)) = best {
                if d_sq <= max_dist_sq {
                    self.follow_id = Some(id);
                }
            }
        } else if input.mouse_pressed(1) {
            let mouse = world_mouse();
            let id = NEXT_BODY_ID.fetch_add(1, Ordering::Relaxed);
            let (mut min_mass, mut max_mass) = (self.mass_min_milli, self.mass_max_milli);
            if min_mass > max_mass {
                std::mem::swap(&mut min_mass, &mut max_mass);
            }
            let (mut min_diam, mut max_diam) = (self.diam_min_milli, self.diam_max_milli);
            if min_diam > max_diam {
                std::mem::swap(&mut min_diam, &mut max_diam);
            }

            let mass = if self.spawn_randomize {
                let lo = (min_mass as f32) / 1000.0;
                let hi = (max_mass as f32) / 1000.0;
                if lo >= hi {
                    lo.max(f32::MIN_POSITIVE)
                } else {
                    (lo + fastrand::f32() * (hi - lo)).max(f32::MIN_POSITIVE)
                }
            } else {
                1.0
            };

            let radius = if self.link_radius_to_mass {
                mass.cbrt()
            } else if self.spawn_randomize {
                let lo = (min_diam as f32) / 1000.0;
                let hi = (max_diam as f32) / 1000.0;
                let diam = if lo >= hi { lo } else { lo + fastrand::f32() * (hi - lo) };
                (diam * 0.5).max(0.01)
            } else {
                1.0
            };

            self.spawn_body = Some(Body::new(id, mouse, Vec2::zero(), mass, radius));
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
                if self.link_radius_to_mass {
                    body.radius = body.mass.cbrt();
                }
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
                std::mem::swap(&mut self.bond_lines, &mut BOND_LINES.lock());
                self.metrics = *METRICS.lock();
            }
            if let Some(body) = self.confirmed_bodies.take() {
                self.bodies.push(body);
                SPAWN.lock().push(body);
            }
            *lock = false;
        }

        if let Some(id) = self.follow_id {
            if let Some(body) = self.bodies.iter().find(|b| b.id == id) {
                // Keep the followed body centered immediately (no per-step "whiplash").
                self.pos = body.pos;
            } else {
                self.follow_id = None;
            }
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

            if self.show_bonds && !self.bond_lines.is_empty() {
                let color = [0xff, 0x00, 0xff, 0xff];
                let thickness_px = self.bond_line_px.max(1) as f32;
                for &(a, b) in &self.bond_lines {
                    let d = b - a;
                    let len = d.mag();
                    if len <= f32::MIN_POSITIVE {
                        continue;
                    }

                    // Quarkstrom renders 1px lines; fake ~2px thickness by drawing two parallel
                    // segments offset by ~1 pixel in world units.
                    let units_per_pixel = self.scale * 2.0 / self.viewport_height as f32;
                    let n = d / len;
                    let perp = Vec2::new(-n.y, n.x);
                    let off = perp * units_per_pixel * thickness_px;
                    ctx.draw_line(a + off, b + off, color);
                    ctx.draw_line(a - off, b - off, color);
                }
            }

            if let Some(id) = self.follow_id {
                if let Some(body) = self.bodies.iter().find(|b| b.id == id) {
                    ctx.draw_circle(body.pos, body.radius * 2.0, [0xff, 0xff, 0x00, 0x60]);
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

                if ui.checkbox(&mut self.show_bonds, "Show Bonds").changed() {
                    WANT_BONDS.store(self.show_bonds, Ordering::Relaxed);
                }
                if self.show_bonds {
                    let mut max_lines = MAX_BOND_LINES.load(Ordering::Relaxed);
                    ui.add(egui::Slider::new(&mut max_lines, 0..=200000).text("Max Bond Lines"));
                    MAX_BOND_LINES.store(max_lines, Ordering::Relaxed);

                    ui.add(egui::Slider::new(&mut self.bond_line_px, 1..=12).text("Bond Thickness (px)"));
                    BOND_LINE_PX.store(self.bond_line_px, Ordering::Relaxed);
                }

                ui.separator();
                ui.label("Playback");
                ui.add(
                    egui::Slider::new(&mut self.init_particles, 1_000..=2_000_000)
                        .logarithmic(true)
                        .text("Init Particles"),
                );
                INIT_PARTICLES.store(self.init_particles, Ordering::Relaxed);
                if ui.button("Reset Simulation").clicked() {
                    RESET_REQUESTED.store(true, Ordering::Relaxed);
                }
                if ui
                    .checkbox(&mut self.collisions_enabled, "Collisions Enabled")
                    .changed()
                {
                    COLLISIONS_ENABLED.store(self.collisions_enabled, Ordering::Relaxed);
                }
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.click_mode_follow, "Left Click: Follow");
                    if ui.button("Unfollow").clicked() {
                        self.follow_id = None;
                    }
                });
                ui.add(
                    egui::Slider::new(&mut self.frame_delay_ms, 0..=500)
                        .text("Frame Delay (ms)"),
                );
                FRAME_DELAY_MS.store(self.frame_delay_ms, Ordering::Relaxed);

                ui.separator();
                ui.label("Particles");
                ui.checkbox(&mut self.spawn_randomize, "Randomize Spawn");
                SPAWN_RANDOMIZE.store(self.spawn_randomize, Ordering::Relaxed);
                ui.checkbox(&mut self.init_randomize, "Randomize On Reset");
                INIT_RANDOMIZE.store(self.init_randomize, Ordering::Relaxed);

                ui.checkbox(&mut self.link_radius_to_mass, "Link Radius to Mass");
                LINK_RADIUS_TO_MASS.store(self.link_radius_to_mass, Ordering::Relaxed);

                ui.add(egui::Slider::new(&mut self.mass_min_milli, 1..=1_000_000).text("Mass Min (milli)"));
                ui.add(egui::Slider::new(&mut self.mass_max_milli, 1..=1_000_000).text("Mass Max (milli)"));
                MASS_MIN_MILLI.store(self.mass_min_milli, Ordering::Relaxed);
                MASS_MAX_MILLI.store(self.mass_max_milli, Ordering::Relaxed);
                ui.label(format!(
                    "mass range [{:.3}, {:.3}]",
                    self.mass_min_milli.min(self.mass_max_milli) as f32 / 1000.0,
                    self.mass_min_milli.max(self.mass_max_milli) as f32 / 1000.0
                ));

                ui.add(
                    egui::Slider::new(&mut self.diam_min_milli, 10..=50_000).text("Diameter Min (milli)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.diam_max_milli, 10..=50_000).text("Diameter Max (milli)"),
                );
                DIAM_MIN_MILLI.store(self.diam_min_milli, Ordering::Relaxed);
                DIAM_MAX_MILLI.store(self.diam_max_milli, Ordering::Relaxed);
                ui.label(format!(
                    "diam range [{:.3}, {:.3}]",
                    self.diam_min_milli.min(self.diam_max_milli) as f32 / 1000.0,
                    self.diam_min_milli.max(self.diam_max_milli) as f32 / 1000.0
                ));

                ui.separator();
                ui.label("Collision Velocity Filter");
                ui.add(
                    egui::Slider::new(&mut self.vel_filter_alpha_milli, 0..=1000)
                        .text("Alpha (milli, 1000=no filter)"),
                );
                VEL_FILTER_ALPHA_MILLI.store(self.vel_filter_alpha_milli, Ordering::Relaxed);
                ui.checkbox(&mut self.vel_filter_contacts_only, "Contacts Only");
                VEL_FILTER_CONTACTS_ONLY.store(self.vel_filter_contacts_only, Ordering::Relaxed);
                ui.add(egui::Slider::new(&mut self.vel_filter_min_contacts, 0..=32).text("Min Contacts"));
                VEL_FILTER_MIN_CONTACTS.store(self.vel_filter_min_contacts, Ordering::Relaxed);
                ui.label(format!(
                    "alpha {:.3}",
                    self.vel_filter_alpha_milli as f32 / 1000.0
                ));

                ui.separator();
                ui.label("Clump Damping (low-speed contacts)");
                ui.checkbox(&mut self.clump_damp_enabled, "Enabled");
                CLUMP_DAMP_ENABLED.store(self.clump_damp_enabled, Ordering::Relaxed);
                ui.checkbox(&mut self.clump_damp_contacts_only, "Contacts Only");
                CLUMP_DAMP_CONTACTS_ONLY.store(self.clump_damp_contacts_only, Ordering::Relaxed);
                ui.add(
                    egui::Slider::new(&mut self.clump_damp_min_contacts, 0..=32)
                        .text("Min Contacts"),
                );
                CLUMP_DAMP_MIN_CONTACTS.store(self.clump_damp_min_contacts, Ordering::Relaxed);
                ui.add(
                    egui::Slider::new(&mut self.clump_damp_speed_milli, 0..=20_000)
                        .text("Speed Threshold (milli units/s)"),
                );
                CLUMP_DAMP_SPEED_MILLI.store(self.clump_damp_speed_milli, Ordering::Relaxed);
                ui.label(format!(
                    "threshold {:.3} units/s",
                    self.clump_damp_speed_milli as f32 / 1000.0
                ));
                ui.add(
                    egui::Slider::new(&mut self.clump_damp_e_low_milli, 0..=1000)
                        .text("Restitution Low (milli)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.clump_damp_e_high_milli, 0..=1000)
                        .text("Restitution High (milli)"),
                );
                CLUMP_DAMP_E_LOW_MILLI.store(self.clump_damp_e_low_milli, Ordering::Relaxed);
                CLUMP_DAMP_E_HIGH_MILLI.store(self.clump_damp_e_high_milli, Ordering::Relaxed);
                ui.label(format!(
                    "e [{:.3}, {:.3}]",
                    self.clump_damp_e_low_milli.min(self.clump_damp_e_high_milli) as f32 / 1000.0,
                    self.clump_damp_e_low_milli.max(self.clump_damp_e_high_milli) as f32 / 1000.0
                ));

                ui.separator();
                ui.label("Clumps");
                if ui
                    .checkbox(&mut self.bonds_enabled, "Rigid Bonds (stabilize clumps)")
                    .changed()
                {
                    BONDS_ENABLED.store(self.bonds_enabled, Ordering::Relaxed);
                }
                ui.add(
                    egui::Slider::new(&mut self.bond_after_frames, 1..=240)
                        .text("Bond After (frames)"),
                );
                BOND_AFTER_FRAMES.store(self.bond_after_frames, Ordering::Relaxed);

                ui.add(
                    egui::Slider::new(&mut self.max_bonds_per_body, 0..=16)
                        .text("Max Bonds / Body"),
                );
                MAX_BONDS_PER_BODY.store(self.max_bonds_per_body, Ordering::Relaxed);

                ui.add(egui::Slider::new(&mut self.bond_iters, 0..=8).text("Bond Iters"));
                BOND_ITERS.store(self.bond_iters, Ordering::Relaxed);

                ui.add(
                    egui::Slider::new(&mut self.bond_break_speed_milli, 0..=20000)
                        .text("Break Speed"),
                );
                ui.label(format!(
                    "{:.3} units/s",
                    self.bond_break_speed_milli as f32 / 1000.0
                ));
                BOND_BREAK_SPEED.store(self.bond_break_speed_milli, Ordering::Relaxed);

                ui.add(
                    egui::Slider::new(&mut self.bond_break_error_milli, 0..=50000)
                        .text("Break Error"),
                );
                ui.label(format!(
                    "{:.3} units",
                    self.bond_break_error_milli as f32 / 1000.0
                ));
                BOND_BREAK_ERROR.store(self.bond_break_error_milli, Ordering::Relaxed);

                ui.collapsing("Fusion (deprecated)", |ui| {
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
                });

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

                    ui.label("collision islands");
                    ui.label(format!("{}", c_last.collision_islands));
                    ui.label(format!("{}", c_avg.collision_islands));
                    ui.end_row();

                    ui.label("max island pairs");
                    ui.label(format!("{}", c_last.collision_island_max_pairs));
                    ui.label(format!("{}", c_avg.collision_island_max_pairs));
                    ui.end_row();

                    ui.label("bonds");
                    ui.label(format!("{}", c_last.bonds));
                    ui.label(format!("{}", c_avg.bonds));
                    ui.end_row();
                });
            });
    }
}
