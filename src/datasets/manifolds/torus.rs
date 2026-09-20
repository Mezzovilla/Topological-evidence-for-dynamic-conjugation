//! Uniform sampling on the standard 2-dimensional ring torus embedded in `R^3`.

use rand::Rng;

use crate::PointCloud;
use crate::datasets::{DatasetError, PointCloudGenerator};

/// The standard 2-dimensional ring torus embedded in `R^3`.
///
/// # Definition
///
/// The torus is the surface of revolution obtained by rotating a circle of
/// radius `r` (the *minor radius*) in the `xz`-plane, centered at
/// `(R, 0, 0)`, about the `z`-axis, where `R` (the *major radius*) is the
/// distance from the center of the tube to the center of the hole. It admits
/// the parametrization
///
/// ```text
/// x = (R + r cos v) cos u,
/// y = (R + r cos v) sin u,
/// z = r sin v,
/// ```
///
/// with angles `u, v in [0, 2*pi)`. Equivalently, its points satisfy the
/// implicit equation
///
/// ```text
/// (sqrt(x^2 + y^2) - R)^2 + z^2 = r^2.
/// ```
///
/// # Dimensions
///
/// * Intrinsic dimension: `2` (a surface, genus 1).
/// * Ambient dimension: `3` (embedded in `R^3`).
///
/// # Parameters
///
/// * `major_radius` (`R`): radius of the central circle swept by the tube
///   center; must be finite and positive.
/// * `minor_radius` (`r`): radius of the tube; must be finite, positive, and
///   strictly smaller than `major_radius` (`R > r`), so the surface is a
///   genuine ring torus (no self-intersection, nonvanishing tube).
///
/// # Probability distribution
///
/// Sampling is **uniform with respect to the intrinsic surface-area measure**
/// of the torus, i.e. the Riemannian volume induced by the embedding in
/// `R^3`. This is *not* the same as sampling the angles `(u, v)` uniformly:
/// the area element is `r (R + r cos v) du dv`, so the outer equator
/// (`v = 0`) carries more area per unit `v` than the inner equator
/// (`v = pi`). Uniform-angle sampling would over-concentrate points near the
/// hole; this generator instead draws `v` with density proportional to
/// `R + r cos v`, which is the correct marginal under surface-area
/// uniformity.
///
/// # Algorithm
///
/// For each point:
///
/// 1. Draw `u ~ Unif[0, 2*pi)` (the conditional distribution of `u` given
///    `v` is uniform by rotational symmetry).
/// 2. Draw `v` by rejection sampling: propose `v ~ Unif[0, 2*pi)`, accept
///    with probability `(R + r cos v) / (R + r)`. Since `0 < R - r <=
///    R + r cos v <= R + r`, the acceptance probability lies in `(0, 1]`
///    and the loop is bounded-free (expected acceptance `R / (R + r)`,
///    always `> 1/2`). Accepted `v` has density proportional to
///    `R + r cos v`, the correct surface-area marginal.
/// 3. Emit `(x, y, z)` via the parametrization above.
///
/// # Example
///
/// ```rust
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use stattda::datasets::{PointCloudGenerator, Torus};
///
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
/// let torus = Torus::new(2.0, 1.0).unwrap();
/// let cloud = torus.sample(&mut rng, 1_000);
/// assert_eq!(cloud.n_points, 1_000);
/// assert_eq!(cloud.ambient_dim, 3);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Torus {
    major_radius: f64,
    minor_radius: f64,
}

impl Torus {
    /// Create a ring torus with major radius `major_radius` (`R`) and minor
    /// radius `minor_radius` (`r`).
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] unless both radii are finite and positive
    /// and `major_radius > minor_radius` (a genuine ring torus requires
    /// `R > r`; `R <= r` gives a horn or spindle torus with a singular or
    /// self-intersecting surface).
    pub fn new(major_radius: f64, minor_radius: f64) -> Result<Self, DatasetError> {
        if !major_radius.is_finite() {
            return Err(DatasetError::NonFiniteParameter {
                generator: "Torus",
                parameter: "major_radius",
                value: major_radius,
            });
        }
        if major_radius <= 0.0 {
            return Err(DatasetError::NonPositiveParameter {
                generator: "Torus",
                parameter: "major_radius",
                value: major_radius,
            });
        }
        if !minor_radius.is_finite() {
            return Err(DatasetError::NonFiniteParameter {
                generator: "Torus",
                parameter: "minor_radius",
                value: minor_radius,
            });
        }
        if minor_radius <= 0.0 {
            return Err(DatasetError::NonPositiveParameter {
                generator: "Torus",
                parameter: "minor_radius",
                value: minor_radius,
            });
        }
        if major_radius <= minor_radius {
            return Err(DatasetError::TorusMajorNotGreaterThanMinor {
                major_radius,
                minor_radius,
            });
        }
        Ok(Self {
            major_radius,
            minor_radius,
        })
    }

    /// Major radius `R`: distance from the torus center to the tube center.
    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Minor radius `r`: radius of the tube.
    pub fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    /// Intrinsic dimension of the torus as a manifold: always `2`.
    pub fn intrinsic_dimension(&self) -> usize {
        2
    }

    /// Dimension of the ambient space the torus is embedded in: always `3`.
    pub fn ambient_dimension(&self) -> usize {
        3
    }
}

impl PointCloudGenerator for Torus {
    fn intrinsic_dimension(&self) -> usize {
        Torus::intrinsic_dimension(self)
    }

    fn ambient_dimension(&self) -> usize {
        Torus::ambient_dimension(self)
    }

    /// Sample `n` points uniformly with respect to the intrinsic
    /// surface-area measure of the torus (see the type-level documentation
    /// for the distribution and the rejection-sampling algorithm).
    ///
    /// `n == 0` yields an empty [`PointCloud`] with `n_points == 0` and
    /// `ambient_dim == 3`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> PointCloud {
        let two_pi = 2.0 * std::f64::consts::PI;
        let r = self.major_radius;
        let minor = self.minor_radius;
        // Acceptance bound: R + r cos v <= R + r, so
        // p_accept(v) = (R + r cos v) / (R + r) in (0, 1].
        let max_area_factor = r + minor;

        let mut coordinates = Vec::with_capacity(n * 3);
        for _ in 0..n {
            let u = rng.random::<f64>() * two_pi;
            // Rejection sampling for v ~ density proportional to
            // R + r cos v on [0, 2*pi). Each iteration accepts with
            // probability R / (R + r) > 1/2, so the loop terminates
            // almost surely and quickly.
            let v = loop {
                let proposal = rng.random::<f64>() * two_pi;
                let accept_prob = (r + minor * proposal.cos()) / max_area_factor;
                if rng.random::<f64>() < accept_prob {
                    break proposal;
                }
            };
            let tube_center_dist = r + minor * v.cos();
            coordinates.push(tube_center_dist * u.cos());
            coordinates.push(tube_center_dist * u.sin());
            coordinates.push(minor * v.sin());
        }

        PointCloud {
            coordinates,
            n_points: n,
            ambient_dim: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    const TOL: f64 = 1e-12;

    /// Implicit defining equation of the torus:
    /// `(sqrt(x^2 + y^2) - R)^2 + z^2 = r^2`. Returns the residual
    /// `|lhs - r^2|` for a single point.
    fn equation_residual(p: &[f64], major: f64, minor: f64) -> f64 {
        let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
        ((rho - major).powi(2) + p[2] * p[2] - minor * minor).abs()
    }

    #[test]
    fn rejects_non_positive_or_non_finite_radii() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(Torus::new(bad, 1.0).is_err(), "major={bad}");
            assert!(Torus::new(2.0, bad).is_err(), "minor={bad}");
        }
    }

    #[test]
    fn rejects_minor_not_smaller_than_major() {
        assert!(Torus::new(1.0, 1.0).is_err());
        assert!(Torus::new(1.0, 2.0).is_err());
        assert!(Torus::new(2.0, 1.999999).is_ok());
    }

    #[test]
    fn zero_samples_gives_empty_cloud_with_ambient_dim_3() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let torus = Torus::new(2.0, 1.0).unwrap();
        let cloud = torus.sample(&mut rng, 0);
        assert_eq!(cloud.n_points, 0);
        assert_eq!(cloud.ambient_dim, 3);
        assert!(cloud.coordinates.is_empty());
    }

    #[test]
    fn sample_count_and_dimensions() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let torus = Torus::new(3.0, 0.5).unwrap();
        assert_eq!(torus.intrinsic_dimension(), 2);
        assert_eq!(torus.ambient_dimension(), 3);
        let cloud = torus.sample(&mut rng, 257);
        assert_eq!(cloud.n_points, 257);
        assert_eq!(cloud.ambient_dim, 3);
        assert_eq!(cloud.coordinates.len(), 257 * 3);
        assert!(cloud.coordinates.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn points_satisfy_defining_equation() {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let (major, minor) = (2.5, 0.75);
        let torus = Torus::new(major, minor).unwrap();
        let cloud = torus.sample(&mut rng, 500);
        for p in cloud.coordinates.chunks_exact(3) {
            assert!(
                equation_residual(p, major, minor) < TOL,
                "point {p:?} violates torus equation"
            );
            // Also check the parametrization ranges directly:
            // rho = sqrt(x^2+y^2) must lie in [R - r, R + r].
            let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(rho >= major - minor - TOL);
            assert!(rho <= major + minor + TOL);
            assert!(p[2].abs() <= minor + TOL);
        }
    }

    #[test]
    fn same_seed_gives_identical_cloud() {
        let torus = Torus::new(2.0, 1.0).unwrap();
        let mut rng_a = ChaCha8Rng::seed_from_u64(99);
        let mut rng_b = ChaCha8Rng::seed_from_u64(99);
        let a = torus.sample(&mut rng_a, 100);
        let b = torus.sample(&mut rng_b, 100);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_give_different_clouds() {
        let torus = Torus::new(2.0, 1.0).unwrap();
        let mut rng_a = ChaCha8Rng::seed_from_u64(1);
        let mut rng_b = ChaCha8Rng::seed_from_u64(2);
        assert_ne!(torus.sample(&mut rng_a, 50), torus.sample(&mut rng_b, 50));
    }

    /// Surface-area uniformity implies `E[cos v] = r / (2R)`: under density
    /// proportional to `R + r cos v`,
    /// `E[cos v] = (int cos v (R + r cos v) dv) / (int (R + r cos v) dv)
    /// = (r pi) / (2 pi R) = r / (2R)`. A parameter-uniform sampler would
    /// give `E[cos v] = 0`, so this test distinguishes the two distributions
    /// (documented acceptance semantics: the outer equator `v = 0` is
    /// over-sampled relative to the inner equator `v = pi` compared with
    /// uniform angles).
    #[test]
    fn v_marginal_matches_surface_area_density() {
        let mut rng = ChaCha8Rng::seed_from_u64(2024);
        let (major, minor) = (2.0, 1.0);
        let torus = Torus::new(major, minor).unwrap();
        let cloud = torus.sample(&mut rng, 50_000);

        // Recover cos v = z / r... no: z = r sin v. Recover cos v from
        // rho: rho = R + r cos v  =>  cos v = (rho - R) / r.
        let n = cloud.n_points as f64;
        let mean_cos_v: f64 = cloud
            .coordinates
            .chunks_exact(3)
            .map(|p| ((p[0] * p[0] + p[1] * p[1]).sqrt() - major) / minor)
            .sum::<f64>()
            / n;
        let expected = minor / (2.0 * major); // 0.25
        assert!(
            (mean_cos_v - expected).abs() < 0.02,
            "E[cos v] = {mean_cos_v}, expected ≈ {expected} (parameter-uniform would be 0)"
        );
    }
}
