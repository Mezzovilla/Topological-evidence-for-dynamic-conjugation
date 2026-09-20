//! Klein bottle generator: point clouds sampled from the standard `R^3`
//! immersion of the Klein bottle.
//!
//! # Geometry
//!
//! The Klein bottle is the non-orientable closed surface obtained as the
//! quotient of the square `[0, 2pi) x [0, 2pi)` under the identifications
//! `(u, 0) ~ (u, 2pi)` and `(0, v) ~ (2pi, 2pi - v)` — one pair of opposite
//! edges glued with a twist. It cannot be embedded in `R^3`; the classical
//! "figure-8" / "bottle" parametrization used here is an *immersion* with a
//! circle of self-intersection where the tube passes through its own neck.
//!
//! The sampled parameter domain is the product `[0, 2pi) x [0, 2pi)` with
//! `(u, v)` drawn independently and uniformly. Under this parametrization the
//! immersion is
//!
//! ```text
//! q = R + r * (cos(u/2) * sin(v) - sin(u/2) * sin(2v))
//! x = q * cos(u)
//! y = q * sin(u)
//! z = r * (sin(u/2) * sin(v) + cos(u/2) * sin(2v))
//! ```
//!
//! where `u` runs around the central "spine" circle of radius `R` and `v`
//! parameterizes the tube cross-section, which rotates by `u/2` as it
//! travels — the half-twist is what produces the non-orientable identification
//! after one full turn in `u`.
//!
//! # Dimensions
//!
//! - Intrinsic dimension: `2` (a closed surface).
//! - Ambient dimension: `3` (immersed in `R^3`, with self-intersections).
//!
//! # Parameters
//!
//! - `major_radius` (`R`): radius of the central circle the tube follows.
//!   Must be finite and strictly positive.
//! - `tube_radius` (`r`): radius of the tube cross-section. Must be finite and
//!   strictly positive.
//!
//! Unlike an embedded torus, this immersion self-intersects for *every*
//! choice of `R` and `r` (that is intrinsic to the Klein bottle in `R^3`), so
//! no `R > r` restriction is imposed; the formula itself places no
//! constraint relating the two radii.
//!
//! # Sampling
//!
//! `u` and `v` are drawn independently and uniformly on `[0, 2pi)` and mapped
//! through the parametrization above. This is uniform in the *parameter
//! domain*, **not** uniform with respect to the surface's intrinsic area
//! measure: the Jacobian of the immersion varies with `(u, v)`, so sampled
//! density on the image is non-uniform. Requiring intrinsic-area-uniform
//! samples would need rejection or importance reweighting by the surface
//! element, which is deliberately out of scope.
//!
//! # Algorithm
//!
//! For each of `n` requested points:
//!
//! 1. Draw `u = 2pi * Uniform(0, 1)` and `v = 2pi * Uniform(0, 1)` from the
//!    caller-supplied RNG.
//! 2. Evaluate the immersion formula to obtain `(x, y, z)`.
//! 3. Append the coordinates row-major to the output [`crate::PointCloud`].
//!
//! `sample(.., 0)` returns an empty cloud with `ambient_dim == 3`.
//!
//! # Example
//!
//! ```
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha8Rng;
//! use stattda::datasets::PointCloudGenerator;
//! use stattda::datasets::manifolds::KleinBottle;
//!
//! let bottle = KleinBottle::new(2.0, 0.8).unwrap();
//! let mut rng = ChaCha8Rng::seed_from_u64(42);
//! let cloud = bottle.sample(&mut rng, 128);
//! assert_eq!(cloud.n_points, 128);
//! assert_eq!(cloud.ambient_dim, 3);
//! ```

use rand::Rng;

use crate::PointCloud;
use crate::datasets::{DatasetError, PointCloudGenerator};

/// Full angular period of both parameters, `2 * pi`.
const TAU: f64 = std::f64::consts::TAU;

/// Point-cloud generator for the standard `R^3` immersion of the Klein
/// bottle. See the module documentation for the parametrization, the
/// quotient/parameter domain, and the sampling-uniformity caveats.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KleinBottle {
    /// Radius `R` of the central circle the tube follows; finite and `> 0`.
    major_radius: f64,
    /// Radius `r` of the tube cross-section; finite and `> 0`.
    tube_radius: f64,
}

impl KleinBottle {
    /// Create a generator with central-circle radius `major_radius` (`R`) and
    /// tube radius `tube_radius` (`r`).
    ///
    /// Both must be finite and strictly positive. No `R > r` constraint is
    /// required: this immersion self-intersects regardless of the radii, and
    /// the formula is well-defined for all positive values.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if either radius is non-finite, zero, or
    /// negative.
    pub fn new(major_radius: f64, tube_radius: f64) -> Result<Self, DatasetError> {
        check_radius("major_radius", major_radius)?;
        check_radius("tube_radius", tube_radius)?;
        Ok(Self {
            major_radius,
            tube_radius,
        })
    }

    /// Radius `R` of the central circle the tube follows.
    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Radius `r` of the tube cross-section.
    pub fn tube_radius(&self) -> f64 {
        self.tube_radius
    }

    /// Evaluate the immersion at parameter point `(u, v)`.
    ///
    /// `u` and `v` are interpreted modulo `2pi`; callers pass values already
    /// drawn on `[0, 2pi)`.
    fn immerse(&self, u: f64, v: f64) -> [f64; 3] {
        let (half_u_sin, half_u_cos) = (u * 0.5).sin_cos();
        let (v_sin, v_cos) = v.sin_cos();
        let sin_2v = 2.0 * v_sin * v_cos;

        let q = self.major_radius + self.tube_radius * (half_u_cos * v_sin - half_u_sin * sin_2v);
        let (u_sin, u_cos) = u.sin_cos();
        let x = q * u_cos;
        let y = q * u_sin;
        let z = self.tube_radius * (half_u_sin * v_sin + half_u_cos * sin_2v);
        [x, y, z]
    }
}

/// Validate one radius argument for [`KleinBottle::new`].
///
/// Reports the dedicated shared variants: [`DatasetError::NonFiniteParameter`]
/// for NaN/infinite values (checked first, so NaN never falls through to the
/// positivity check) and [`DatasetError::NonPositiveParameter`] for finite
/// values `<= 0`, consistent with `Sphere`/`Torus`.
fn check_radius(name: &'static str, value: f64) -> Result<(), DatasetError> {
    if !value.is_finite() {
        return Err(DatasetError::NonFiniteParameter {
            generator: "KleinBottle",
            parameter: name,
            value,
        });
    }
    if value <= 0.0 {
        return Err(DatasetError::NonPositiveParameter {
            generator: "KleinBottle",
            parameter: name,
            value,
        });
    }
    Ok(())
}

impl PointCloudGenerator for KleinBottle {
    /// Intrinsic dimension of the Klein bottle: `2` (a closed surface).
    fn intrinsic_dimension(&self) -> usize {
        2
    }

    /// Ambient dimension of the immersion: `3`.
    fn ambient_dimension(&self) -> usize {
        3
    }

    /// Draw `n` parameter-uniform samples.
    ///
    /// `u` and `v` are independent `Uniform[0, 2pi)` draws per point, mapped
    /// through the Klein bottle immersion. `n == 0` yields an empty cloud with
    /// `ambient_dim == 3`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> PointCloud {
        let ambient_dim = self.ambient_dimension();
        let mut coordinates = Vec::with_capacity(n * ambient_dim);
        for _ in 0..n {
            let u = rng.random_range(0.0..TAU);
            let v = rng.random_range(0.0..TAU);
            coordinates.extend_from_slice(&self.immerse(u, v));
        }
        PointCloud {
            coordinates,
            n_points: n,
            ambient_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    const R: f64 = 2.0;
    const TUBE: f64 = 0.8;
    const TOL: f64 = 1e-12;

    fn bottle() -> KleinBottle {
        KleinBottle::new(R, TUBE).unwrap()
    }

    /// Direct, deliberately un-vectorized evaluation of the documented
    /// parametrization, used to cross-check [`KleinBottle::immerse`].
    fn reference_immerse(major_radius: f64, tube_radius: f64, u: f64, v: f64) -> [f64; 3] {
        let q = major_radius
            + tube_radius * ((u / 2.0).cos() * v.sin() - (u / 2.0).sin() * (2.0 * v).sin());
        [
            q * u.cos(),
            q * u.sin(),
            tube_radius * ((u / 2.0).sin() * v.sin() + (u / 2.0).cos() * (2.0 * v).sin()),
        ]
    }

    /// Invert the immersion locally: recover `(u, v)` from a point produced by
    /// [`KleinBottle::immerse`] with `u, v in [0, 2pi)`.
    ///
    /// `u` is recovered from `atan2(y, x)`. For `v`, rotate the tube-frame
    /// coordinates `a = (q - R) / r`, `b = z / r` by `-u/2` to undo the
    /// half-twist, giving `sin v` directly; the two `asin` branches are then
    /// disambiguated by re-immersing both candidates and keeping the closer
    /// reconstruction.
    fn inverse(kb: &KleinBottle, point: [f64; 3]) -> (f64, f64) {
        let [x, y, z] = point;
        let u = y.atan2(x).rem_euclid(TAU);
        let q = (x * x + y * y).sqrt();
        // a = (q - R) / r =  cos(u/2) sin v - sin(u/2) sin 2v
        // b = z / r       =  sin(u/2) sin v + cos(u/2) sin 2v
        let a = (q - kb.major_radius) / kb.tube_radius;
        let b = z / kb.tube_radius;
        let (hs, hc) = (u * 0.5).sin_cos();
        let sin_v = hc * a + hs * b;
        let v_asin = sin_v.asin();
        let mut best_v = 0.0;
        let mut best_err = f64::INFINITY;
        for cand in [
            v_asin.rem_euclid(TAU),
            (std::f64::consts::PI - v_asin).rem_euclid(TAU),
        ] {
            let p = kb.immerse(u, cand);
            let err = (p[0] - x).abs() + (p[1] - y).abs() + (p[2] - z).abs();
            if err < best_err {
                best_err = err;
                best_v = cand;
            }
        }
        (u, best_v)
    }

    #[test]
    fn rejects_invalid_radii() {
        // NaN is reported as NonFiniteParameter; compare fields rather than
        // the whole error because NaN != NaN under PartialEq.
        for (parameter, err) in [
            (
                "major_radius",
                KleinBottle::new(f64::NAN, TUBE).unwrap_err(),
            ),
            ("tube_radius", KleinBottle::new(R, f64::NAN).unwrap_err()),
        ] {
            match err {
                DatasetError::NonFiniteParameter {
                    generator,
                    parameter: p,
                    value,
                } => {
                    assert_eq!(generator, "KleinBottle");
                    assert_eq!(p, parameter);
                    assert!(value.is_nan());
                }
                other => panic!("expected NonFiniteParameter, got {other:?}"),
            }
        }
        for bad in [f64::INFINITY, f64::NEG_INFINITY] {
            for (parameter, err) in [
                ("major_radius", KleinBottle::new(bad, TUBE).unwrap_err()),
                ("tube_radius", KleinBottle::new(R, bad).unwrap_err()),
            ] {
                assert_eq!(
                    err,
                    DatasetError::NonFiniteParameter {
                        generator: "KleinBottle",
                        parameter,
                        value: bad,
                    },
                    "{parameter} = {bad}"
                );
            }
        }
        for bad in [0.0, -1.0] {
            assert_eq!(
                KleinBottle::new(bad, TUBE).unwrap_err(),
                DatasetError::NonPositiveParameter {
                    generator: "KleinBottle",
                    parameter: "major_radius",
                    value: bad,
                }
            );
            assert_eq!(
                KleinBottle::new(R, bad).unwrap_err(),
                DatasetError::NonPositiveParameter {
                    generator: "KleinBottle",
                    parameter: "tube_radius",
                    value: bad,
                }
            );
        }
        assert!(KleinBottle::new(R, TUBE).is_ok());
    }

    #[test]
    fn getters_return_radii() {
        let kb = bottle();
        assert_eq!(kb.major_radius(), R);
        assert_eq!(kb.tube_radius(), TUBE);
    }

    #[test]
    fn dimensions_are_surface_in_r3() {
        let kb = bottle();
        assert_eq!(kb.intrinsic_dimension(), 2);
        assert_eq!(kb.ambient_dimension(), 3);
    }

    #[test]
    fn zero_samples_empty_cloud() {
        let kb = bottle();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let cloud = kb.sample(&mut rng, 0);
        assert_eq!(cloud.n_points, 0);
        assert_eq!(cloud.ambient_dim, 3);
        assert!(cloud.coordinates.is_empty());
    }

    #[test]
    fn sample_counts_and_dims() {
        let kb = bottle();
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let n = 257;
        let cloud = kb.sample(&mut rng, n);
        assert_eq!(cloud.n_points, n);
        assert_eq!(cloud.ambient_dim, 3);
        assert_eq!(cloud.coordinates.len(), n * 3);
        assert!(cloud.coordinates.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn immerse_matches_reference_formula() {
        let kb = bottle();
        // Deterministic grid over the parameter domain, including edge angles.
        for i in 0..17 {
            let u = (i as f64) * TAU / 17.0;
            for j in 0..9 {
                let v = (j as f64) * TAU / 9.0;
                let got = kb.immerse(u, v);
                let want = reference_immerse(R, TUBE, u, v);
                for k in 0..3 {
                    assert!(
                        (got[k] - want[k]).abs() <= TOL,
                        "u={u} v={v} coord {k}: {got:?} vs {want:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn immersion_round_trips_through_inverse() {
        let kb = bottle();
        for i in 0..24 {
            let u = (i as f64) * TAU / 24.0 + 0.01;
            let v = ((i as f64) * 0.37 * TAU).rem_euclid(TAU);
            let p = kb.immerse(u, v);
            let (u2, v2) = inverse(&kb, p);
            let p2 = kb.immerse(u2, v2);
            for k in 0..3 {
                assert!(
                    (p[k] - p2[k]).abs() <= TOL,
                    "reconstruction drift for u={u} v={v}: {p:?} vs {p2:?}"
                );
            }
        }
    }

    #[test]
    fn same_seed_identical_clouds() {
        let kb = bottle();
        let mut rng1 = ChaCha8Rng::seed_from_u64(0xC0FFEE);
        let mut rng2 = ChaCha8Rng::seed_from_u64(0xC0FFEE);
        let c1 = kb.sample(&mut rng1, 64);
        let c2 = kb.sample(&mut rng2, 64);
        assert_eq!(c1, c2);
    }

    #[test]
    fn different_seeds_differ() {
        let kb = bottle();
        let mut rng1 = ChaCha8Rng::seed_from_u64(1);
        let mut rng2 = ChaCha8Rng::seed_from_u64(2);
        let c1 = kb.sample(&mut rng1, 64);
        let c2 = kb.sample(&mut rng2, 64);
        assert_ne!(c1, c2);
    }
}
