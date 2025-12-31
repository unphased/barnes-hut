use std::sync::atomic::Ordering;

mod body;
mod metrics;
mod partition;
mod quadtree;
mod renderer;
mod simulation;
mod utils;

use renderer::Renderer;
use simulation::Simulation;

fn main() {
    let threads = std::thread::available_parallelism().unwrap().get().max(3) - 2;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let config = quarkstrom::Config {
        window_mode: quarkstrom::WindowMode::Windowed(900, 900),
    };

    let mut simulation = Simulation::new();
    renderer::NEXT_BODY_ID.store(simulation.bodies.len() as u64, Ordering::Relaxed);

    std::thread::spawn(move || {
	    loop {
            if renderer::RESET_REQUESTED.swap(false, Ordering::Relaxed) {
                simulation = Simulation::new();
                renderer::NEXT_BODY_ID.store(simulation.bodies.len() as u64, Ordering::Relaxed);
                renderer::SPAWN.lock().clear();
            }

	        if renderer::PAUSED.load(Ordering::Relaxed) {
	            std::thread::yield_now();
	        } else {
	            simulation.step();
	        }
	        render(&mut simulation);

            let delay_ms = renderer::FRAME_DELAY_MS.load(Ordering::Relaxed);
            if delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(delay_ms));
            }
	    }
    });

    quarkstrom::run::<Renderer>(config);
}

fn render(simulation: &mut Simulation) {
    let mut lock = renderer::UPDATE_LOCK.lock();
    for body in renderer::SPAWN.lock().drain(..) {
        simulation.bodies.push(body);
    }
    {
        let mut lock = renderer::BODIES.lock();
        lock.clear();
        lock.extend_from_slice(&simulation.bodies);
    }
    {
        let mut lock = renderer::QUADTREE.lock();
        lock.clear();
        if renderer::WANT_QUADTREE.load(Ordering::Relaxed) {
            lock.extend_from_slice(&simulation.quadtree.nodes);
        }
    }
    {
        let mut lock = renderer::METRICS.lock();
        *lock = simulation.metrics.snapshot();
    }
    *lock |= true;
}
