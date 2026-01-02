use crate::{body::Body, metrics::Metrics, quadtree::Quadtree, utils};

use broccoli::aabb::Rect;
use broccoli_rayon::{build::RayonBuildPar, prelude::RayonQueryPar};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use ultraviolet::Vec2;

#[derive(Clone, Copy, Debug, Default)]
struct Contact {
    last_frame: usize,
    streak: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Bond {
    rest_len: f32,
}

pub struct Simulation {
    pub dt: f32,
    pub frame: usize,
    pub bodies: Vec<Body>,
    pub quadtree: Quadtree,
    pub metrics: Metrics,
    contacts: HashMap<(u64, u64), Contact>,
    bonds: HashMap<(u64, u64), Bond>,
    vel_prev: Vec<Vec2>,
}

impl Simulation {
    pub fn new(n: usize) -> Self {
        let dt = 0.05;
        let theta = 1.0;
        let epsilon = 1.0;
        let leaf_capacity = 16;
        let thread_capacity = 1024;

        let init_randomize =
            crate::renderer::INIT_RANDOMIZE.load(std::sync::atomic::Ordering::Relaxed);
        let link_radius_to_mass =
            crate::renderer::LINK_RADIUS_TO_MASS.load(std::sync::atomic::Ordering::Relaxed);
        let mass_min =
            crate::renderer::MASS_MIN_MILLI.load(std::sync::atomic::Ordering::Relaxed) as f32
                / 1000.0;
        let mass_max =
            crate::renderer::MASS_MAX_MILLI.load(std::sync::atomic::Ordering::Relaxed) as f32
                / 1000.0;
        let diam_min =
            crate::renderer::DIAM_MIN_MILLI.load(std::sync::atomic::Ordering::Relaxed) as f32
                / 1000.0;
        let diam_max =
            crate::renderer::DIAM_MAX_MILLI.load(std::sync::atomic::Ordering::Relaxed) as f32
                / 1000.0;

        let randomize = init_randomize.then_some(utils::RandomizeConfig {
            mass_min,
            mass_max,
            diam_min,
            diam_max,
            link_radius_to_mass,
        });

        let bodies: Vec<Body> = utils::uniform_disc(n, randomize);
        let quadtree = Quadtree::new(theta, epsilon, leaf_capacity, thread_capacity);

        Self {
            dt,
            frame: 0,
            bodies,
            quadtree,
            metrics: Metrics::default(),
            contacts: HashMap::new(),
            bonds: HashMap::new(),
            vel_prev: Vec::new(),
        }
    }

    pub fn step(&mut self) {
        self.metrics.begin_frame(self.frame);

        let start = Instant::now();
        self.iterate();
        self.metrics.snapshot_mut().last.iterate = start.elapsed();

        let start = Instant::now();
        let collision_pairs = self.collide();
        self.metrics.snapshot_mut().last.collide = start.elapsed();
        self.metrics.snapshot_mut().counts_last.collision_pairs = collision_pairs;

        let start = Instant::now();
        self.attract();
        self.metrics.snapshot_mut().last.attract_total = start.elapsed();

        self.metrics.snapshot_mut().counts_last.bodies = self.bodies.len();
        self.metrics.snapshot_mut().counts_last.quadtree_nodes_active = self.quadtree.active_nodes_len();
        self.metrics.snapshot_mut().counts_last.bonds = self.bonds.len();
        self.metrics.finish_frame();
        self.frame += 1;
    }

    pub fn bond_lines(&self, max_lines: usize) -> Vec<(Vec2, Vec2)> {
        if max_lines == 0 || self.bonds.is_empty() || self.bodies.is_empty() {
            return Vec::new();
        }

        let max_id = self.bodies.iter().map(|b| b.id).max().unwrap_or(0) as usize;
        let use_vec = max_id <= self.bodies.len().saturating_mul(4);

        let mut out = Vec::with_capacity(self.bonds.len().min(max_lines));

        if use_vec {
            let mut id_to_pos = vec![Vec2::zero(); max_id.saturating_add(1)];
            for body in &self.bodies {
                let idx = body.id as usize;
                if idx < id_to_pos.len() {
                    id_to_pos[idx] = body.pos;
                }
            }

            for (&(a, b), _) in self.bonds.iter() {
                if out.len() >= max_lines {
                    break;
                }
                let ai = a as usize;
                let bi = b as usize;
                if ai >= id_to_pos.len() || bi >= id_to_pos.len() {
                    continue;
                }
                out.push((id_to_pos[ai], id_to_pos[bi]));
            }
        } else {
            let mut id_to_pos = HashMap::<u64, Vec2>::with_capacity(self.bodies.len());
            for body in &self.bodies {
                id_to_pos.insert(body.id, body.pos);
            }

            for (&(a, b), _) in self.bonds.iter() {
                if out.len() >= max_lines {
                    break;
                }
                let (Some(&p1), Some(&p2)) = (id_to_pos.get(&a), id_to_pos.get(&b)) else {
                    continue;
                };
                out.push((p1, p2));
            }
        }

        out
    }

    pub fn attract(&mut self) {
        let start = Instant::now();
        self.quadtree.build(&mut self.bodies);
        self.metrics.snapshot_mut().last.quadtree_build = start.elapsed();

        let start = Instant::now();
        self.quadtree.acc(&mut self.bodies);
        self.metrics.snapshot_mut().last.quadtree_acc = start.elapsed();
    }

    pub fn iterate(&mut self) {
        for body in &mut self.bodies {
            body.update(self.dt);
        }
    }

    pub fn collide(&mut self) -> usize {
        // Snapshot velocities before we apply any collision/bond response this frame. This makes the
        // "velocity filter" act on collision-induced changes rather than damping gravity/integration.
        if self.vel_prev.len() != self.bodies.len() {
            self.vel_prev.resize(self.bodies.len(), Vec2::zero());
        }
        for (i, body) in self.bodies.iter().enumerate() {
            self.vel_prev[i] = body.vel;
        }

        let mut rects = self
            .bodies
            .iter()
            .enumerate()
            .map(|(index, body)| {
                let pos = body.pos;
                let radius = body.radius;
                let min = pos - Vec2::one() * radius;
                let max = pos + Vec2::one() * radius;
                (Rect::new(min.x, max.x, min.y, max.y), index)
            })
            .collect::<Vec<_>>();

        let mut broccoli = broccoli::Tree::par_new(&mut rects);

        let bodies_ptr = self.bodies.as_ptr() as usize;
        let bodies_len = self.bodies.len();
        let pairs = broccoli.par_find_colliding_pairs_acc_closure(
            Vec::<(usize, usize)>::new(),
            |_| Vec::<(usize, usize)>::new(),
            |acc, mut b| acc.append(&mut b),
            |acc, i, j| {
                let i = *i.unpack_inner();
                let j = *j.unpack_inner();

                let bodies = unsafe { std::slice::from_raw_parts(bodies_ptr as *const Body, bodies_len) };
                let b1 = bodies[i];
                let b2 = bodies[j];
                let d = b2.pos - b1.pos;
                let r = b1.radius + b2.radius;
                if d.mag_sq() <= r * r {
                    acc.push((i, j));
                }
            },
        );

        let frame = self.frame;

        let vel_alpha_milli = crate::renderer::VEL_FILTER_ALPHA_MILLI
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(1000);
        let vel_alpha = vel_alpha_milli as f32 / 1000.0;
        let vel_contacts_only =
            crate::renderer::VEL_FILTER_CONTACTS_ONLY.load(std::sync::atomic::Ordering::Relaxed);
        let vel_min_contacts = crate::renderer::VEL_FILTER_MIN_CONTACTS
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(1024) as u16;

        let clump_damp_enabled =
            crate::renderer::CLUMP_DAMP_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let clump_damp_contacts_only =
            crate::renderer::CLUMP_DAMP_CONTACTS_ONLY.load(std::sync::atomic::Ordering::Relaxed);
        let clump_damp_min_contacts = crate::renderer::CLUMP_DAMP_MIN_CONTACTS
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(1024) as u16;
        let clump_speed = crate::renderer::CLUMP_DAMP_SPEED_MILLI
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(1_000_000) as f32
            / 1000.0;
        let mut e_low = crate::renderer::CLUMP_DAMP_E_LOW_MILLI
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(1000) as f32
            / 1000.0;
        let mut e_high = crate::renderer::CLUMP_DAMP_E_HIGH_MILLI
            .load(std::sync::atomic::Ordering::Relaxed)
            .min(1000) as f32
            / 1000.0;
        if e_low > e_high {
            std::mem::swap(&mut e_low, &mut e_high);
        }

        let need_contact_counts = (vel_contacts_only && vel_alpha_milli < 1000)
            || (clump_damp_enabled && clump_damp_contacts_only);
        let mut contact_counts: Vec<u16> = if need_contact_counts {
            vec![0; self.bodies.len()]
        } else {
            Vec::new()
        };

        let bond_enabled =
            crate::renderer::BONDS_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let bond_steps = crate::renderer::BOND_AFTER_FRAMES
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(1) as u32;
        let bond_break_speed = crate::renderer::BOND_BREAK_SPEED
            .load(std::sync::atomic::Ordering::Relaxed) as f32
            / 1000.0;
        let max_bonds_per_body = crate::renderer::MAX_BONDS_PER_BODY
            .load(std::sync::atomic::Ordering::Relaxed)
            .clamp(0, 16) as u8;
        let bond_iters = crate::renderer::BOND_ITERS
            .load(std::sync::atomic::Ordering::Relaxed)
            .clamp(0, 16) as u32;

        let fuse_enabled =
            crate::renderer::FUSE_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let fuse_steps = crate::renderer::FUSE_AFTER_FRAMES
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(1) as u32;

        // Track persistent contacts (used for both fusion and bonds).
        let mut fuse_set = HashSet::<(u64, u64)>::new();
        let mut fuse_pairs = Vec::<(usize, usize)>::new();
        let mut bond_candidates = Vec::<(usize, usize)>::new();

        // Avoid tracking every pair in dense clumps: only track a limited number of contacts per
        // body per frame for streak accounting (enough to form a sparse bond graph).
        let track_contacts = fuse_enabled || (bond_enabled && max_bonds_per_body > 0);
        let mut contact_budget = if track_contacts {
            let max_id = self.bodies.iter().map(|b| b.id).max().unwrap_or(0) as usize;
            vec![0u8; max_id.saturating_add(1)]
        } else {
            Vec::new()
        };
        let per_body_contact_limit = (max_bonds_per_body.saturating_mul(2)).clamp(2, 8);
        for &(i, j) in &pairs {
            let a = self.bodies[i].id;
            let b = self.bodies[j].id;
            let key = if a < b { (a, b) } else { (b, a) };

            if track_contacts {
                let ai = a as usize;
                let bi = b as usize;
                if ai < contact_budget.len() && bi < contact_budget.len() {
                    if contact_budget[ai] >= per_body_contact_limit
                        || contact_budget[bi] >= per_body_contact_limit
                    {
                        continue;
                    }
                    contact_budget[ai] += 1;
                    contact_budget[bi] += 1;
                }
            }

            let entry = self.contacts.entry(key).or_default();
            entry.streak = if entry.last_frame + 1 == frame {
                entry.streak.saturating_add(1)
            } else {
                1
            };
            entry.last_frame = frame;

            if fuse_enabled && entry.streak >= fuse_steps {
                fuse_set.insert(key);
                fuse_pairs.push((i, j));
            }
            if bond_enabled && entry.streak >= bond_steps {
                bond_candidates.push((i, j));
            }
        }

        // Keep contact state bounded (drop pairs not seen recently).
        if frame % 30 == 0 {
            self.contacts
                .retain(|_, c| frame.saturating_sub(c.last_frame) <= 1);
        }

        if !contact_counts.is_empty() {
            for &(i, j) in &pairs {
                contact_counts[i] = contact_counts[i].saturating_add(1);
                contact_counts[j] = contact_counts[j].saturating_add(1);
            }
        }

        // Add bonds, capped per body.
        if bond_enabled && max_bonds_per_body > 0 && !bond_candidates.is_empty() {
            let mut bond_count = HashMap::<u64, u8>::new();
            for &(a, b) in self.bonds.keys() {
                *bond_count.entry(a).or_insert(0) += 1;
                *bond_count.entry(b).or_insert(0) += 1;
            }

            for (i, j) in bond_candidates {
                let a = self.bodies[i].id;
                let b = self.bodies[j].id;
                if a == b {
                    continue;
                }
                let key = if a < b { (a, b) } else { (b, a) };
                if self.bonds.contains_key(&key) {
                    continue;
                }
                if bond_count.get(&a).copied().unwrap_or(0) >= max_bonds_per_body {
                    continue;
                }
                if bond_count.get(&b).copied().unwrap_or(0) >= max_bonds_per_body {
                    continue;
                }

                let d = self.bodies[j].pos - self.bodies[i].pos;
                let min_len = self.bodies[i].radius + self.bodies[j].radius;
                let rest_len = d.mag().max(min_len);
                self.bonds.insert(
                    key,
                    Bond { rest_len },
                );
                *bond_count.entry(a).or_insert(0) += 1;
                *bond_count.entry(b).or_insert(0) += 1;
            }
        }

        // Resolve collisions (skip pairs fused or directly bonded).
        let mut bonds_to_break = Vec::<(u64, u64)>::new();
        for &(i, j) in &pairs {
            let a = self.bodies[i].id;
            let b = self.bodies[j].id;
            let key = if a < b { (a, b) } else { (b, a) };

            if fuse_enabled && fuse_set.contains(&key) {
                continue;
            }

            let restitution = if clump_damp_enabled && clump_speed > 0.0 {
                let p1 = self.bodies[i].pos;
                let p2 = self.bodies[j].pos;
                let d = p2 - p1;
                let len = d.mag();
                let rel_v = self.bodies[j].vel - self.bodies[i].vel;
                let rel_n = if len > f32::MIN_POSITIVE {
                    (rel_v.dot(d / len)).abs()
                } else {
                    rel_v.mag()
                };
                let in_clump = !clump_damp_contacts_only
                    || (!contact_counts.is_empty()
                        && contact_counts[i] >= clump_damp_min_contacts
                        && contact_counts[j] >= clump_damp_min_contacts);

                if in_clump && rel_n < clump_speed {
                    let t = (rel_n / clump_speed).clamp(0.0, 1.0);
                    e_low + (e_high - e_low) * t
                } else {
                    e_high
                }
            } else {
                e_high
            };

            if bond_enabled {
                if self.bonds.contains_key(&key) {
                    let rel_v = (self.bodies[j].vel - self.bodies[i].vel).mag();
                    if bond_break_speed > 0.0 && rel_v >= bond_break_speed {
                        bonds_to_break.push(key);
                        self.resolve_with_restitution(i, j, e_high);
                    }
                    continue;
                }
            }

            self.resolve_with_restitution(i, j, restitution);
        }

        for key in bonds_to_break {
            self.bonds.remove(&key);
            self.contacts.remove(&key);
        }

        if fuse_enabled && !fuse_pairs.is_empty() {
            self.fuse_contacts(fuse_pairs);
        }

        // Project bonds to stabilize clumps.
        if bond_enabled && bond_iters > 0 && !self.bonds.is_empty() {
            self.solve_bonds(bond_iters);
        }

        self.apply_velocity_filter(vel_alpha, vel_contacts_only, vel_min_contacts, &contact_counts);

        pairs.len()
    }

    fn apply_velocity_filter(
        &mut self,
        alpha: f32,
        contacts_only: bool,
        min_contacts: u16,
        contact_counts: &[u16],
    ) {
        // Filter is applied as a scale on the collision-induced velocity change:
        //   v <- v_before_collisions + alpha * (v_after - v_before_collisions)
        // This avoids canceling gravity when the filter is applied globally.
        let alpha = alpha.clamp(0.0, 1.0);
        for (index, body) in self.bodies.iter_mut().enumerate() {
            let base = self.vel_prev.get(index).copied().unwrap_or(body.vel);

            if contacts_only && !contact_counts.is_empty() && contact_counts[index] < min_contacts {
                continue;
            }

            let cur = body.vel;
            if alpha >= 1.0 {
                continue;
            }
            if alpha <= 0.0 {
                body.vel = base;
                continue;
            }

            let next = base + (cur - base) * alpha;
            body.vel = next;
        }
    }

    fn solve_bonds(&mut self, iters: u32) {
        let break_error =
            crate::renderer::BOND_BREAK_ERROR.load(std::sync::atomic::Ordering::Relaxed) as f32
                / 1000.0;

        let max_id = self.bodies.iter().map(|b| b.id).max().unwrap_or(0) as usize;
        let mut id_to_index = vec![usize::MAX; max_id.saturating_add(1)];
        for (index, body) in self.bodies.iter().enumerate() {
            id_to_index[body.id as usize] = index;
        }

        let mut break_list = Vec::<(u64, u64)>::new();

        for _ in 0..iters {
            for (&(a, b), bond) in self.bonds.iter_mut() {
                let ai = a as usize;
                let bi = b as usize;
                if ai >= id_to_index.len() || bi >= id_to_index.len() {
                    break_list.push((a, b));
                    continue;
                }
                let i = id_to_index[ai];
                let j = id_to_index[bi];
                if i == usize::MAX || j == usize::MAX {
                    break_list.push((a, b));
                    continue;
                }

                let p1 = self.bodies[i].pos;
                let p2 = self.bodies[j].pos;
                let d = p2 - p1;
                let len = d.mag();
                if len <= f32::MIN_POSITIVE {
                    continue;
                }
                let err = len - bond.rest_len;
                if break_error > 0.0 && err.abs() >= break_error {
                    break_list.push((a, b));
                    continue;
                }

                let n = d / len;
                let w1 = 1.0 / self.bodies[i].mass.max(f32::MIN_POSITIVE);
                let w2 = 1.0 / self.bodies[j].mass.max(f32::MIN_POSITIVE);
                let w_sum = w1 + w2;
                if w_sum <= 0.0 {
                    continue;
                }

                let corr = n * (err / w_sum);
                self.bodies[i].pos += corr * w1;
                self.bodies[j].pos -= corr * w2;

                // Mild velocity update to reduce jitter.
                let dv = corr / self.dt * 0.25;
                self.bodies[i].vel += dv;
                self.bodies[j].vel -= dv;
            }
        }

        for key in break_list {
            self.bonds.remove(&key);
            self.contacts.remove(&key);
        }
    }

    fn resolve_with_restitution(&mut self, i: usize, j: usize, restitution: f32) {
        let b1 = &self.bodies[i];
        let b2 = &self.bodies[j];

        let p1 = b1.pos;
        let p2 = b2.pos;

        let r1 = b1.radius;
        let r2 = b2.radius;

        let d = p2 - p1;
        let r = r1 + r2;

        if d.mag_sq() > r * r {
            return;
        }

        let v1 = b1.vel;
        let v2 = b2.vel;

        let v = v2 - v1;

        let d_dot_v = d.dot(v);

        let m1 = b1.mass;
        let m2 = b2.mass;

        let weight1 = m2 / (m1 + m2);
        let weight2 = m1 / (m1 + m2);

        if d_dot_v >= 0.0 && d != Vec2::zero() {
            let tmp = d * (r / d.mag() - 1.0);
            self.bodies[i].pos -= weight1 * tmp;
            self.bodies[j].pos += weight2 * tmp;
            return;
        }

        let v_sq = v.mag_sq();
        let d_sq = d.mag_sq();
        let r_sq = r * r;

        let t = (d_dot_v + (d_dot_v * d_dot_v - v_sq * (d_sq - r_sq)).max(0.0).sqrt()) / v_sq;

        self.bodies[i].pos -= v1 * t;
        self.bodies[j].pos -= v2 * t;

        let p1 = self.bodies[i].pos;
        let p2 = self.bodies[j].pos;
        let d = p2 - p1;
        let d_dot_v = d.dot(v);
        let d_sq = d.mag_sq();

        let restitution = restitution.clamp(0.0, 1.0);
        let tmp = d * ((1.0 + restitution) * d_dot_v / d_sq);
        let v1 = v1 + tmp * weight1;
        let v2 = v2 - tmp * weight2;

        self.bodies[i].vel = v1;
        self.bodies[j].vel = v2;
        self.bodies[i].pos += v1 * t;
        self.bodies[j].pos += v2 * t;
    }

    fn fuse_contacts(&mut self, pairs: Vec<(usize, usize)>) {
        if pairs.is_empty() || self.bodies.is_empty() {
            return;
        }

        let mut involved = Vec::<usize>::new();
        involved.reserve(pairs.len() * 2);
        for (i, j) in &pairs {
            involved.push(*i);
            involved.push(*j);
        }
        involved.sort_unstable();
        involved.dedup();
        if involved.len() <= 1 {
            return;
        }

        let mut index_to_local = HashMap::<usize, usize>::with_capacity(involved.len());
        for (local, &index) in involved.iter().enumerate() {
            index_to_local.insert(index, local);
        }

        let mut uf = UnionFind::new(involved.len());
        for (i, j) in pairs {
            let (Some(&li), Some(&lj)) = (index_to_local.get(&i), index_to_local.get(&j)) else {
                continue;
            };
            uf.union(li, lj);
        }

        #[derive(Clone, Copy, Default)]
        struct Agg {
            mass: f32,
            pos_m: Vec2,
            vel_m: Vec2,
            id_min: u64,
        }

        let mut agg = HashMap::<usize, Agg>::new();
        for (local, &index) in involved.iter().enumerate() {
            let root = uf.find(local);
            let body = self.bodies[index];
            let entry = agg.entry(root).or_insert(Agg {
                mass: 0.0,
                pos_m: Vec2::zero(),
                vel_m: Vec2::zero(),
                id_min: body.id,
            });
            entry.mass += body.mass;
            entry.pos_m += body.pos * body.mass;
            entry.vel_m += body.vel * body.mass;
            entry.id_min = entry.id_min.min(body.id);
        }

        let mut removed = vec![false; self.bodies.len()];
        for &index in &involved {
            if let Some(flag) = removed.get_mut(index) {
                *flag = true;
            }
        }

        let mut new_bodies = Vec::with_capacity(self.bodies.len() - involved.len() + agg.len());
        for (index, body) in self.bodies.iter().copied().enumerate() {
            if !removed[index] {
                new_bodies.push(body);
            }
        }

        for (_, a) in agg {
            let inv_m = 1.0 / a.mass.max(f32::MIN_POSITIVE);
            let pos = a.pos_m * inv_m;
            let vel = a.vel_m * inv_m;
            let mass = a.mass;
            let radius = mass.cbrt();
            new_bodies.push(Body::new(a.id_min, pos, vel, mass, radius));
        }

        self.bodies = new_bodies;
    }
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let parent = self.parent[x];
        if parent == x {
            return x;
        }
        let root = self.find(parent);
        self.parent[x] = root;
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut a = self.find(a);
        let mut b = self.find(b);
        if a == b {
            return;
        }
        let ra = self.rank[a];
        let rb = self.rank[b];
        if ra < rb {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        if ra == rb {
            self.rank[a] = ra.saturating_add(1);
        }
    }
}
