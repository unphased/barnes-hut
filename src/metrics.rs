use std::collections::VecDeque;
use std::time::Duration;

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameTimings {
    pub iterate: Duration,
    pub collide: Duration,
    pub attract_total: Duration,
    pub quadtree_build: Duration,
    pub quadtree_acc: Duration,
}

impl FrameTimings {
    pub fn total(&self) -> Duration {
        self.iterate + self.collide + self.attract_total
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameCounts {
    pub bodies: usize,
    pub quadtree_nodes_active: usize,
    pub collision_pairs: usize,
    pub collision_islands: usize,
    pub collision_island_max_pairs: usize,
    pub bonds: usize,
}

impl FrameCounts {
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetricsSnapshot {
    pub frame: usize,
    pub last: FrameTimings,
    pub avg: FrameTimings,
    pub counts_last: FrameCounts,
    pub counts_avg: FrameCounts,
}

#[derive(Debug)]
pub struct Metrics {
    window: usize,
    frames: VecDeque<FrameTimings>,
    counts: VecDeque<FrameCounts>,
    sum: FrameTimings,
    sum_counts: FrameCounts,
    snapshot: MetricsSnapshot,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new(120)
    }
}

impl Metrics {
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            frames: VecDeque::new(),
            counts: VecDeque::new(),
            sum: FrameTimings::default(),
            sum_counts: FrameCounts::default(),
            snapshot: MetricsSnapshot::default(),
        }
    }

    pub fn begin_frame(&mut self, frame: usize) {
        self.snapshot.frame = frame;
        self.snapshot.last.clear();
        self.snapshot.counts_last.clear();
    }

    pub fn finish_frame(&mut self) {
        let last = self.snapshot.last;
        let last_counts = self.snapshot.counts_last;

        self.frames.push_back(last);
        self.counts.push_back(last_counts);

        self.sum.iterate += last.iterate;
        self.sum.collide += last.collide;
        self.sum.attract_total += last.attract_total;
        self.sum.quadtree_build += last.quadtree_build;
        self.sum.quadtree_acc += last.quadtree_acc;

        self.sum_counts.bodies += last_counts.bodies;
        self.sum_counts.quadtree_nodes_active += last_counts.quadtree_nodes_active;
        self.sum_counts.collision_pairs += last_counts.collision_pairs;
        self.sum_counts.collision_islands += last_counts.collision_islands;
        self.sum_counts.collision_island_max_pairs += last_counts.collision_island_max_pairs;
        self.sum_counts.bonds += last_counts.bonds;

        while self.frames.len() > self.window {
            if let Some(old) = self.frames.pop_front() {
                self.sum.iterate -= old.iterate;
                self.sum.collide -= old.collide;
                self.sum.attract_total -= old.attract_total;
                self.sum.quadtree_build -= old.quadtree_build;
                self.sum.quadtree_acc -= old.quadtree_acc;
            }
            if let Some(old) = self.counts.pop_front() {
                self.sum_counts.bodies -= old.bodies;
                self.sum_counts.quadtree_nodes_active -= old.quadtree_nodes_active;
                self.sum_counts.collision_pairs -= old.collision_pairs;
                self.sum_counts.collision_islands -= old.collision_islands;
                self.sum_counts.collision_island_max_pairs -= old.collision_island_max_pairs;
                self.sum_counts.bonds -= old.bonds;
            }
        }

        let denom = self.frames.len().max(1) as u32;
        self.snapshot.avg.iterate = self.sum.iterate / denom;
        self.snapshot.avg.collide = self.sum.collide / denom;
        self.snapshot.avg.attract_total = self.sum.attract_total / denom;
        self.snapshot.avg.quadtree_build = self.sum.quadtree_build / denom;
        self.snapshot.avg.quadtree_acc = self.sum.quadtree_acc / denom;

        let denom_counts = self.counts.len().max(1) as f32;
        self.snapshot.counts_avg.bodies = (self.sum_counts.bodies as f32 / denom_counts) as usize;
        self.snapshot.counts_avg.quadtree_nodes_active =
            (self.sum_counts.quadtree_nodes_active as f32 / denom_counts) as usize;
        self.snapshot.counts_avg.collision_pairs =
            (self.sum_counts.collision_pairs as f32 / denom_counts) as usize;
        self.snapshot.counts_avg.collision_islands =
            (self.sum_counts.collision_islands as f32 / denom_counts) as usize;
        self.snapshot.counts_avg.collision_island_max_pairs =
            (self.sum_counts.collision_island_max_pairs as f32 / denom_counts) as usize;
        self.snapshot.counts_avg.bonds = (self.sum_counts.bonds as f32 / denom_counts) as usize;
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        self.snapshot
    }

    pub fn snapshot_mut(&mut self) -> &mut MetricsSnapshot {
        &mut self.snapshot
    }
}

pub fn duration_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}
