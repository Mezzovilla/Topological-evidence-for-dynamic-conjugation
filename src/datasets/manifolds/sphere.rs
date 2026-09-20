//! Uniform sampling on the `n`-sphere.

use rand::Rng;

use crate::PointCloud;
use crate::datasets::{DatasetError, PointCloudGenerator};

/// Generator of points sampled uniformly from the `n`-sphere.
///
/// # Mathematical definition
///
/// The `n`-sphere of radius `r > 0` centred at the origin of `R^{n+1}` is
///
/// ```text
/// S^n(r) = { x in R^{n+1} : ||x||_2 = r }.
/// ```
///
/// [`Sphere::new`] takes the *intrinsic* dimension `n >= 0` and the radius;
/// `n = 0` is the two-point set `{-r, +r}` in `R^1`, `n = 1` is a circle in
/// `R^2`, `n = 2` is the ordinary sphere in `R^3`, and so on.
///
/// # Parameters
///
/// * `intrinsic_dimension` — the manifold dimension `n` (`usize`, `>= 0`); the
///   ambient dimension is `n + 1` and overflow of `n + 1` is reported as
///   [`DatasetError::SphereDimensionOverflow`].
/// * `radius` — finite and strictly positive; NaN/infinite values give
///   [`DatasetError::NonFiniteParameter`] and values `<= 0` give
///   [`DatasetError::NonPositiveParameter`].
///
/// # Distribution
///
/// Points are uniform on `S^n(r)` with respect to the intrinsic volume
/// measure (the `n`-dimensional Hausdorff/surface measure induced by the
/// embedding): the probability of landing in a measurable patch is
/// proportional to its `n`-dimensional surface area.
///
/// # Algorithm
///
/// Each point draws `n + 1` independent standard-normal coordinates via
/// Box–Muller applied to `rng`-generated uniforms, then rescales the vector to
/// norm `r`. A Gaussian vector's direction is uniform on `S^n(1)` by
/// rotational invariance, so radial rescaling gives the target distribution.
/// The all-zero draw (a measure-zero event) is resampled rather than
/// normalised. `n == 0` therefore yields `{-r, +r}` with equal probability.
///
/// # Example
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use stattda::datasets::{PointCloudGenerator, Sphere};
///
/// let sphere = Sphere::new(2, 1.5).unwrap();
/// assert_eq!(sphere.intrinsic_dimension(), 2);
/// assert_eq!(sphere.ambient_dimension(), 3);
///
/// let mut rng = ChaCha8Rng::seed_from_u64(7);
/// let cloud = sphere.sample(&mut rng, 100);
/// assert_eq!(cloud.n_points, 100);
/// assert_eq!(cloud.ambient_dim, 3);
/// for p in 0..cloud.n_points {
///     let row = &cloud.coordinates[p * 3..(p + 1) * 3];
///     let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
///     assert!((norm - 1.5).abs() < 1e-12);
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    intrinsic_dimension: usize,
    radius: f64,
}

impl Sphere {
    /// Construct a generator for the `n`-sphere of the given radius.
    ///
    /// See the [type-level documentation](Sphere) for parameter semantics and
    /// the returned [`DatasetError`] variants.
    pub fn new(intrinsic_dimension: usize, radius: f64) -> Result<Self, DatasetError> {
        intrinsic_dimension
            .checked_add(1)
            .ok_or(DatasetError::SphereDimensionOverflow(intrinsic_dimension))?;
        if !radius.is_finite() {
            return Err(DatasetError::NonFiniteParameter {
                generator: "Sphere",
                parameter: "radius",
                value: radius,
            });
        }
        if radius <= 0.0 {
            return Err(DatasetError::NonPositiveParameter {
                generator: "Sphere",
                parameter: "radius",
                value: radius,
            });
        }
        Ok(Sphere {
            intrinsic_dimension,
            radius,
        })
    }

    /// Intrinsic dimension `n` of the sphere.
    pub fn intrinsic_dimension(&self) -> usize {
        self.intrinsic_dimension
    }

    /// Radius `r` of the sphere.
    pub fn radius(&self) -> f64 {
        self.radius
    }
}

/// Draw one standard-normal coordinate via Box–Muller.
///
/// Uses `sqrt(-2 ln u1) * cos(2 pi u2)`; only the cosine output is consumed
/// (one RNG call per coordinate keeps the algorithm simple and stable, at the
/// cost of discarding the sine output).
fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    // `rng.random::<f64>()` is uniform on [0, 1); map 0 to 1 so u in (0, 1]
    // and ln(u) stays finite.
    let u1 = 1.0 - rng.random::<f64>();
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

impl PointCloudGenerator for Sphere {
    fn intrinsic_dimension(&self) -> usize {
        self.intrinsic_dimension
    }

    fn ambient_dimension(&self) -> usize {
        // Safe: `Sphere::new` rejects `intrinsic_dimension == usize::MAX`.
        self.intrinsic_dimension + 1
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> PointCloud {
        let ambient_dim = self.ambient_dimension();
        let mut coordinates = Vec::with_capacity(n * ambient_dim);
        for _ in 0..n {
            // Draw a Gaussian direction; the zero vector is a measure-zero
            // event, resample it rather than producing NaNs.
            let (point, norm) = loop {
                let v: Vec<f64> = (0..ambient_dim).map(|_| standard_normal(rng)).collect();
                let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 {
                    break (v, norm);
                }
            };
            let scale = self.radius / norm;
            coordinates.extend(point.iter().map(|x| x * scale));
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

    #[test]
    fn new_rejects_non_finite_radius() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = Sphere::new(2, bad).unwrap_err();
            // NaN != NaN, so match structurally rather than with assert_eq!.
            match err {
                DatasetError::NonFiniteParameter {
                    generator,
                    parameter,
                    value,
                } => {
                    assert_eq!(generator, "Sphere");
                    assert_eq!(parameter, "radius");
                    assert!(value.is_nan() == bad.is_nan() || value == bad);
                }
                other => panic!("expected NonFiniteParameter, got {other:?}"),
            }
        }
    }

    #[test]
    fn new_rejects_non_positive_radius() {
        for bad in [0.0, -0.0, -1.5] {
            let err = Sphere::new(2, bad).unwrap_err();
            assert_eq!(
                err,
                DatasetError::NonPositiveParameter {
                    generator: "Sphere",
                    parameter: "radius",
                    value: bad,
                }
            );
        }
    }

    #[test]
    fn new_rejects_dimension_overflow() {
        let err = Sphere::new(usize::MAX, 1.0).unwrap_err();
        assert_eq!(err, DatasetError::SphereDimensionOverflow(usize::MAX));
    }

    #[test]
    fn new_accepts_valid_parameters() {
        let s = Sphere::new(3, 2.0).unwrap();
        assert_eq!(s.intrinsic_dimension(), 3);
        assert_eq!(s.radius(), 2.0);
        assert_eq!(s.ambient_dimension(), 4);
    }

    #[test]
    fn zero_samples_empty_cloud_with_correct_dims() {
        let s = Sphere::new(2, 1.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let cloud = s.sample(&mut rng, 0);
        assert_eq!(
            cloud,
            PointCloud {
                coordinates: vec![],
                n_points: 0,
                ambient_dim: 3,
            }
        );
    }

    #[test]
    fn sample_counts_and_dimensions() {
        let s = Sphere::new(4, 1.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let cloud = s.sample(&mut rng, 37);
        assert_eq!(cloud.n_points, 37);
        assert_eq!(cloud.ambient_dim, 5);
        assert_eq!(cloud.coordinates.len(), 37 * 5);
    }

    #[test]
    fn every_sampled_point_has_exact_radius() {
        let radius = 2.75;
        let s = Sphere::new(6, radius).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let cloud = s.sample(&mut rng, 200);
        let d = cloud.ambient_dim;
        for p in 0..cloud.n_points {
            let row = &cloud.coordinates[p * d..(p + 1) * d];
            let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - radius).abs() < 1e-12,
                "point {p} has norm {norm}, expected {radius}"
            );
        }
    }

    #[test]
    fn same_seed_produces_identical_cloud() {
        let s = Sphere::new(2, 1.0).unwrap();
        let mut rng_a = ChaCha8Rng::seed_from_u64(42);
        let mut rng_b = ChaCha8Rng::seed_from_u64(42);
        assert_eq!(s.sample(&mut rng_a, 50), s.sample(&mut rng_b, 50));
    }

    #[test]
    fn different_seeds_produce_different_clouds() {
        let s = Sphere::new(2, 1.0).unwrap();
        let mut rng_a = ChaCha8Rng::seed_from_u64(1);
        let mut rng_b = ChaCha8Rng::seed_from_u64(2);
        assert_ne!(s.sample(&mut rng_a, 10), s.sample(&mut rng_b, 10));
    }

    #[test]
    fn s0_is_the_two_point_set() {
        // S^0 = {-r, +r}: every sample must be one of the two points.
        let s = Sphere::new(0, 3.0).unwrap();
        assert_eq!(s.ambient_dimension(), 1);
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let cloud = s.sample(&mut rng, 100);
        for &x in &cloud.coordinates {
            assert!(
                (x - 3.0).abs() < 1e-12 || (x + 3.0).abs() < 1e-12,
                "S^0 sample {x} not in {{-3, +3}} (to 1e-12)"
            );
        }
        // Both points should occur with this many draws (overwhelmingly
        // probable; structural check only, not a distribution test).
        assert!(cloud.coordinates.iter().any(|&x| x > 0.0));
        assert!(cloud.coordinates.iter().any(|&x| x < 0.0));
    }
}
