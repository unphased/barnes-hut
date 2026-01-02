use crate::body::Body;
use ultraviolet::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct RandomizeConfig {
    pub mass_min: f32,
    pub mass_max: f32,
    pub diam_min: f32,
    pub diam_max: f32,
    pub link_radius_to_mass: bool,
}

pub fn uniform_disc(n: usize, randomize: Option<RandomizeConfig>) -> Vec<Body> {
    fastrand::seed(0);
    let inner_radius = 25.0;
    let outer_radius = (n as f32).sqrt() * 5.0;

    let mut bodies: Vec<Body> = Vec::with_capacity(n);

    let m = 1e6;
    let center = Body::new(0, Vec2::zero(), Vec2::zero(), m as f32, inner_radius);
    bodies.push(center);

    while bodies.len() < n {
        let a = fastrand::f32() * std::f32::consts::TAU;
        let (sin, cos) = a.sin_cos();
        let t = inner_radius / outer_radius;
        let r = fastrand::f32() * (1.0 - t * t) + t * t;
        let pos = Vec2::new(cos, sin) * outer_radius * r.sqrt();
        let vel = Vec2::new(sin, -cos);
        let mut mass = 1.0f32;
        let mut radius = mass.cbrt();

        if let Some(cfg) = randomize {
            let (mut min_mass, mut max_mass) = (cfg.mass_min, cfg.mass_max);
            if min_mass > max_mass {
                std::mem::swap(&mut min_mass, &mut max_mass);
            }
            if min_mass > 0.0 {
                mass = if min_mass >= max_mass {
                    min_mass
                } else {
                    min_mass + fastrand::f32() * (max_mass - min_mass)
                };
            }

            if cfg.link_radius_to_mass {
                radius = mass.cbrt();
            } else {
                let (mut min_d, mut max_d) = (cfg.diam_min, cfg.diam_max);
                if min_d > max_d {
                    std::mem::swap(&mut min_d, &mut max_d);
                }
                let diam = if min_d >= max_d {
                    min_d
                } else {
                    min_d + fastrand::f32() * (max_d - min_d)
                };
                radius = (diam * 0.5).max(0.01);
            }
        }

        bodies.push(Body::new(0, pos, vel, mass, radius));
    }

    bodies.sort_by(|a, b| a.pos.mag_sq().total_cmp(&b.pos.mag_sq()));
    let mut mass = 0.0;
    for i in 0..n {
        mass += bodies[i].mass;
        if bodies[i].pos == Vec2::zero() {
            continue;
        }

        let v = (mass / bodies[i].pos.mag()).sqrt();
        bodies[i].vel *= v;
    }

    for (id, body) in bodies.iter_mut().enumerate() {
        body.id = id as u64;
    }

    bodies
}
