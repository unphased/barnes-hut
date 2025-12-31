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

pub struct Simulation {
    pub dt: f32,
    pub frame: usize,
    pub bodies: Vec<Body>,
    pub quadtree: Quadtree,
    pub metrics: Metrics,
    contacts: HashMap<(u64, u64), Contact>,
}

impl Simulation {
    pub fn new() -> Self {
        let dt = 0.05;
        let n = 200000;
        let theta = 1.0;
        let epsilon = 1.0;
        let leaf_capacity = 16;
        let thread_capacity = 1024;

        let bodies: Vec<Body> = utils::uniform_disc(n);
        let quadtree = Quadtree::new(theta, epsilon, leaf_capacity, thread_capacity);

        Self {
            dt,
            frame: 0,
            bodies,
            quadtree,
            metrics: Metrics::default(),
            contacts: HashMap::new(),
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
        self.metrics.finish_frame();
        self.frame += 1;
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

        let pairs = broccoli.par_find_colliding_pairs_acc_closure(
            Vec::<(usize, usize)>::new(),
            |_| Vec::<(usize, usize)>::new(),
            |acc, mut b| acc.append(&mut b),
            |acc, i, j| {
                let i = *i.unpack_inner();
                let j = *j.unpack_inner();
                acc.push((i, j));
            },
        );

        let frame = self.frame;
        let fuse_enabled =
            crate::renderer::FUSE_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let fuse_steps = crate::renderer::FUSE_AFTER_FRAMES
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(1) as u32;

        let mut fuse_set = HashSet::<(u64, u64)>::new();
        let mut fuse_pairs = Vec::<(usize, usize)>::new();
        for &(i, j) in &pairs {
            let a = self.bodies[i].id;
            let b = self.bodies[j].id;
            let key = if a < b { (a, b) } else { (b, a) };

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
        }

        if frame % 30 == 0 {
            self.contacts
                .retain(|_, c| frame.saturating_sub(c.last_frame) <= 1);
        }

        for &(i, j) in &pairs {
            let a = self.bodies[i].id;
            let b = self.bodies[j].id;
            let key = if a < b { (a, b) } else { (b, a) };
            if !fuse_set.contains(&key) {
                self.resolve(i, j);
            }
        }

        if fuse_enabled && !fuse_pairs.is_empty() {
            self.fuse_contacts(fuse_pairs);
        }

        pairs.len()
    }

    fn resolve(&mut self, i: usize, j: usize) {
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

        let tmp = d * (1.5 * d_dot_v / d_sq);
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
