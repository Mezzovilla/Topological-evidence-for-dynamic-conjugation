//! Synthetic dataset generators for topological data analysis.
//!
//! This module provides generators that draw point clouds from mathematically
//! well-defined spaces — manifolds such as spheres, tori, and Klein bottles —
//! so that TDA pipelines can be exercised on data with known ground-truth
//! topology.
//!
//! All generators implement the [`crate::datasets::PointCloudGenerator`] trait,
//! which exposes the intrinsic and ambient dimensions of the space and a single
//! [`crate::datasets::PointCloudGenerator::sample`] entry point that draws `n`
//! points into a
//! [`crate::PointCloud`] using any caller-supplied [`rand::Rng`]. Supplying a
//! seeded RNG (e.g. `rand_chacha::ChaCha8Rng`) makes every generated dataset
//! exactly reproducible.
//!
//! Construction-time parameter errors (non-finite or non-positive radii,
//! inconsistent torus radii, dimension overflow, …) are reported via
//! [`crate::datasets::DatasetError`]; sampling itself is infallible for a
//! successfully constructed generator.
//!
//! Semantics of "uniform": unless a generator's documentation says otherwise,
//! samples are drawn uniformly with respect to the space's intrinsic volume
//! measure (for a sphere, the Hausdorff/surface measure induced by its
//! embedding). No unit test in this crate claims to *prove* the sampling
//! distribution empirically; tests only check deterministic structural
//! properties (counts, dimensions, exact norms, seed reproducibility).

/// Generators for closed manifolds: spheres, tori, and Klein bottles.
pub mod manifolds;

pub use manifolds::{KleinBottle, Sphere, Torus};

/// Errors returned when constructing a dataset generator with invalid
/// parameters.
///
/// Each variant names the generator and the offending parameter so failures
/// can be matched and reported precisely. Sampling from a successfully
/// constructed generator never produces this error.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum DatasetError {
    /// A radius parameter was NaN, infinite, or otherwise not finite.
    #[error("{generator}: {parameter} must be finite, got {value}")]
    NonFiniteParameter {
        /// Name of the generator (e.g. `"Sphere"`).
        generator: &'static str,
        /// Name of the offending parameter (e.g. `"radius"`).
        parameter: &'static str,
        /// The rejected value.
        value: f64,
    },

    /// A radius or length parameter was finite but `<= 0`.
    #[error("{generator}: {parameter} must be positive, got {value}")]
    NonPositiveParameter {
        /// Name of the generator (e.g. `"Sphere"`).
        generator: &'static str,
        /// Name of the offending parameter (e.g. `"radius"`).
        parameter: &'static str,
        /// The rejected value.
        value: f64,
    },

    /// A torus was constructed with `major_radius <= minor_radius`, which is
    /// not an embedded torus (self-intersecting or degenerate).
    #[error(
        "Torus: major_radius ({major_radius}) must be greater than minor_radius ({minor_radius})"
    )]
    TorusMajorNotGreaterThanMinor {
        /// The rejected major radius.
        major_radius: f64,
        /// The rejected minor radius.
        minor_radius: f64,
    },

    /// A sphere intrinsic dimension `n` would require ambient dimension
    /// `n + 1`, which overflows `usize`.
    #[error("Sphere: intrinsic dimension {0} overflows ambient dimension n + 1")]
    SphereDimensionOverflow(usize),

    /// Generic invalid-parameter fallback for generator-specific constraints
    /// not covered by a dedicated variant. Prefer the specific variants above
    /// when they describe the failure.
    #[error("invalid parameter {parameter}: {reason}")]
    InvalidParameter {
        /// Name of the offending parameter (e.g. `"major_radius"`).
        parameter: &'static str,
        /// Human-readable explanation of the constraint that was violated.
        reason: String,
    },
}

/// A source of point clouds sampled from a fixed space.
///
/// Implementors describe a space through its intrinsic dimension (the
/// dimension of the manifold itself) and its ambient dimension (the dimension
/// of the Euclidean space the samples are embedded in, equal to
/// [`crate::PointCloud::ambient_dim`] of every sampled cloud).
///
/// Sampling is deterministic given the RNG state: two identically seeded RNGs
/// produce identical [`crate::PointCloud`]s.
pub trait PointCloudGenerator {
    /// Intrinsic dimension of the sampled space (e.g. `n` for the `n`-sphere).
    fn intrinsic_dimension(&self) -> usize;

    /// Ambient (embedding) dimension of every sampled point.
    fn ambient_dimension(&self) -> usize;

    /// Draw `n` points from the space into a [`crate::PointCloud`].
    ///
    /// The returned cloud satisfies `n_points == n`,
    /// `ambient_dim == self.ambient_dimension()`, and
    /// `coordinates.len() == n * self.ambient_dimension()`. `n == 0` yields an
    /// empty cloud with the correct `ambient_dim`.
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R, n: usize) -> crate::PointCloud;
}
