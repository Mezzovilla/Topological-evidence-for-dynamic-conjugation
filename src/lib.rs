//! # stattda — Robinson–Turner two-sample test on persistence diagrams
//!
//! This crate implements the two-sample randomization test of Robinson & Turner,
//! *Hypothesis Testing for Topological Data Analysis* (arXiv:1310.7467v2, 2016),
//! Sections 3–4: two independent groups of point clouds are compared through
//! their `k`-dimensional persistence diagrams using the within-group joint-loss
//! statistic `F_{p,q}` and an exact or Monte-Carlo permutation p-value.
//!
//! ## Statistical claim
//!
//! A small p-value is evidence against exchangeability of the group labels for
//! the selected topological summary and preprocessing pipeline. It does **not**
//! prove that an individual point cloud has a significant topological feature,
//! and it does not establish a causal group effect.
//!
//! ## Pipeline
//!
//! Each point cloud is mapped independently to one degree-`k` persistence
//! diagram via a Euclidean Vietoris–Rips filtration over `Z/2` truncated at
//! `max_edge_length`, computed by the pinned `oxicuda-tda = 0.5.5` backend
//! (a pre-1.0, alpha-status crate; all of its types are kept internal).
//! Diagram distances use the optimal `L-infinity` ground metric convention
//!
//! ```text
//! W_p(X, Y) = ( min_phi  sum_x ||x - phi(x)||_inf^p )^(1/p)
//! ```
//!
//! with diagonal matching (`p` is the Wasserstein order; the ground norm is
//! always `L-infinity`), and `W_infinity` (bottleneck) for `p = infinity`.
//!
//! Permutations act at the **cloud level**, never the point level: every cloud
//! is one independent observation producing exactly one diagram, and raw points
//! are never pooled across clouds. The full `N x N` diagram-distance matrix is
//! computed once (upper triangle only, mirrored, exact zero diagonal) and reused
//! for every labeling, so permutation work touches only the matrix.
//!
//! The statistic
//!
//! ```text
//! F_{p,q}(L) = sum_m  (1 / (n_m (n_m - 1)))  sum_{i<j in group m}  W_p(D_i, D_j)^q
//! ```
//!
//! is algebraically equivalent to equation (4) of Robinson & Turner; `p` is the
//! Wasserstein order and `q in {1, 2}` is the loss exponent (default `F_{2,2}`).
//!
//! Because the filtration is finite, degree-`k` classes may be essential (never
//! die). They are surfaced explicitly and handled per
//! [`EssentialClassPolicy`]; note `H_0` almost always needs
//! [`EssentialClassPolicy::Drop`] because one connected component stays
//! essential. Diagrams depend on the filtration and on scale — results describe
//! group differences under *this* pipeline, not filtration-independent
//! topology.
//!
//! Computational scaling: one persistent-homology run per cloud, `O(N^2)`
//! optimal diagram distances, then matrix-only permutation work.
//!
//! ## Example
//!
//! ```no_run
//! use stattda::*;
//!
//! let square = PointCloud::try_from_rows(vec![
//!     vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0],
//! ]).unwrap();
//! let square2 = PointCloud::try_from_rows(vec![
//!     vec![0.1, 0.0], vec![1.0, 0.1], vec![0.9, 1.0], vec![0.0, 0.9],
//! ]).unwrap();
//! let blob = PointCloud::try_from_rows(vec![
//!     vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1], vec![0.1, 0.1],
//! ]).unwrap();
//! let blob2 = PointCloud::try_from_rows(vec![
//!     vec![0.0, 0.0], vec![0.2, 0.0], vec![0.0, 0.2], vec![0.2, 0.2],
//! ]).unwrap();
//!
//! let config = RobinsonTurnerConfig::new(1, 1.5); // homology_dim, max_edge_length
//! let result = robinson_turner_two_sample_test(
//!     &[square, square2],
//!     &[blob, blob2],
//!     &config,
//! ).unwrap();
//! println!("F = {}, p = {}", result.statistic, result.p_value);
//! ```

use std::cell::Cell;

use itertools::Itertools;
use rand::SeedableRng;
use rand::rngs::OsRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use thiserror::Error;

use oxicuda_tda::complex::filtration::Filtration;
use oxicuda_tda::homology::boundary::BoundaryMatrix;
use oxicuda_tda::homology::persistent::{PersistencePair, extract_persistence_pairs};
use oxicuda_tda::homology::reduction::reduce_boundary_matrix;
use oxicuda_tda::persistence::diagram::PersistenceDiagram;
use oxicuda_tda::persistence::distance::bottleneck_distance;
use oxicuda_tda::persistence::wasserstein_p::wasserstein_p;

/// Version string of the exactly-pinned `oxicuda-tda` backend.
const BACKEND_NAME: &str = "oxicuda-tda";
const BACKEND_VERSION: &str = "0.5.5";
/// `oxicuda-tda 0.5.5` supports filtration dimensions through 6, so the
/// requested homology degree is limited to 5 (filtration dim = k + 1).
const MAX_HOMOLOGY_DIM: usize = 5;

// ---------------------------------------------------------------------------
// Public input types
// ---------------------------------------------------------------------------

/// A point cloud in `R^d`, stored row-major (`coordinates.len() == n_points *
/// ambient_dim`).
#[derive(Clone, Debug, PartialEq)]
pub struct PointCloud {
    /// Row-major coordinates: point `i`, coordinate `j` at `i * ambient_dim + j`.
    pub coordinates: Vec<f64>,
    /// Number of points (rows).
    pub n_points: usize,
    /// Ambient dimension (columns); must be positive.
    pub ambient_dim: usize,
}

impl PointCloud {
    /// Build a cloud from a rectangular, nonempty list of rows with finite
    /// coordinates.
    pub fn try_from_rows(rows: Vec<Vec<f64>>) -> Result<Self, RobinsonTurnerError> {
        if rows.is_empty() {
            return Err(RobinsonTurnerError::MalformedCloud {
                index: usize::MAX,
                reason: "cloud has no points".to_string(),
            });
        }
        let ambient_dim = rows[0].len();
        if ambient_dim == 0 {
            return Err(RobinsonTurnerError::MalformedCloud {
                index: 0,
                reason: "ambient dimension is zero".to_string(),
            });
        }
        let mut coordinates = Vec::with_capacity(rows.len() * ambient_dim);
        for (i, row) in rows.iter().enumerate() {
            if row.len() != ambient_dim {
                return Err(RobinsonTurnerError::MalformedCloud {
                    index: i,
                    reason: format!("row has {} coordinates, expected {ambient_dim}", row.len()),
                });
            }
            for (j, &x) in row.iter().enumerate() {
                if !x.is_finite() {
                    return Err(RobinsonTurnerError::NonFiniteCoordinate {
                        index: i,
                        offset: j,
                    });
                }
                coordinates.push(x);
            }
        }
        Ok(PointCloud {
            coordinates,
            n_points: rows.len(),
            ambient_dim,
        })
    }
}

/// Optimal diagram-distance order used in `F_{p,q}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum DiagramDistance {
    /// 1-Wasserstein distance (`L-infinity` ground metric, diagonal matching).
    Wasserstein1,
    /// 2-Wasserstein distance (`L-infinity` ground metric, diagonal matching).
    Wasserstein2,
    /// Bottleneck distance `W_infinity`.
    Bottleneck,
}

/// How the permutation p-value is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum InferenceMethod {
    /// Exact enumeration when `C(N, n1) <= max_exact_labelings`, else Monte Carlo.
    Auto,
    /// Enumerate all `C(N, n1)` allocations; error if that exceeds
    /// `max_exact_labelings`.
    Exact,
    /// Draw `n_permutations` uniform random allocations.
    MonteCarlo,
}

/// Policy for degree-`k` persistence classes that never die within the
/// truncated filtration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum EssentialClassPolicy {
    /// Return [`RobinsonTurnerError::EssentialClasses`] (default).
    Reject,
    /// Remove requested-degree essential classes and record how many were
    /// removed per cloud. Almost always required for `H_0`, where one
    /// connected component stays essential.
    Drop,
}

/// Configuration for [`robinson_turner_two_sample_test`].
///
/// `homology_dim` and `max_edge_length` have no scientifically universal
/// defaults; construct with [`RobinsonTurnerConfig::new`]. [`Default`] uses
/// `homology_dim = 1` and `max_edge_length = 1.0` purely as mechanical
/// placeholders — scientific callers should use `new`.
#[derive(Clone, Debug)]
pub struct RobinsonTurnerConfig {
    /// Requested homology degree `k`; must be `<= 5` (backend filtration limit).
    pub homology_dim: usize,
    /// Finite, strictly positive Rips truncation (max edge length).
    pub max_edge_length: f64,
    /// Diagram-distance order `p`.
    pub diagram_distance: DiagramDistance,
    /// Loss exponent `q in {1, 2}` (default 2 → `F_{p,2}`).
    pub loss_q: u32,
    /// Exact-enumeration guard on `C(N, n1)` (default 1,000,000).
    pub max_exact_labelings: u128,
    /// Monte-Carlo draw count `B` (default 9,999; minimum p-value `0.0001`).
    /// Must be nonzero whenever Monte Carlo is selected.
    pub n_permutations: u64,
    /// Inference method selection (default [`InferenceMethod::Auto`]).
    pub method: InferenceMethod,
    /// Optional `u64` seed for `ChaCha8Rng`. If `None`, a seed is drawn from
    /// `OsRng` and recorded in the result so runs are reproducible.
    pub random_seed: Option<u64>,
    /// Significance level in `(0, 1)` (default 0.05).
    pub alpha: f64,
    /// Losses within this tolerance of the observed statistic count as ties
    /// (conservative: `F_perm <= F_observed + tie_tolerance`). Default 1e-12.
    pub tie_tolerance: f64,
    /// Essential-class handling (default [`EssentialClassPolicy::Reject`]).
    pub essential_class_policy: EssentialClassPolicy,
    /// If true, return the symmetric `N x N` diagram-distance matrix.
    pub return_distance_matrix: bool,
}

impl RobinsonTurnerConfig {
    /// Create a configuration with the two required scientific choices; all
    /// other fields take the documented defaults of [`Default`].
    pub fn new(homology_dim: usize, max_edge_length: f64) -> Self {
        Self {
            homology_dim,
            max_edge_length,
            ..Self::default()
        }
    }
}

impl Default for RobinsonTurnerConfig {
    /// Mechanical defaults only — `homology_dim = 1` and
    /// `max_edge_length = 1.0` are placeholders; scientific callers should use
    /// [`RobinsonTurnerConfig::new`].
    fn default() -> Self {
        Self {
            homology_dim: 1,
            max_edge_length: 1.0,
            diagram_distance: DiagramDistance::Wasserstein2,
            loss_q: 2,
            max_exact_labelings: 1_000_000,
            n_permutations: 9_999,
            method: InferenceMethod::Auto,
            random_seed: None,
            alpha: 0.05,
            tie_tolerance: 1e-12,
            essential_class_policy: EssentialClassPolicy::Reject,
            return_distance_matrix: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Which inference mode was actually used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum InferenceMode {
    /// All `C(N, n1)` allocations were enumerated (including the observed one).
    Exact,
    /// `n_random_permutations` uniform allocations were sampled.
    MonteCarlo,
}

/// Persistent-homology pipeline provenance.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PhConfiguration {
    /// Requested homology degree.
    pub homology_dim: usize,
    /// Rips truncation used for every cloud.
    pub max_edge_length: f64,
    /// Requested maximum simplex dimension (`homology_dim + 1`). Per cloud,
    /// the backend filtration is additionally clamped to `n_points - 1` when
    /// the cloud has too few points to form larger simplices.
    pub max_simplex_dim: usize,
    /// Coefficient field, always `"Z/2"`.
    pub coefficient_field: String,
    /// Filtration description, always Euclidean Vietoris–Rips.
    pub filtration: String,
    /// Essential-class policy applied.
    pub essential_class_policy: EssentialClassPolicy,
}

/// Online summary of the permutation-loss distribution.
///
/// `mean`/`sample_std_dev` use Welford updates; `min`/`max` are exact;
/// `approximate_p25/p50/p75` come from deterministic P² quantile estimators and
/// are only approximate — the p-value never depends on them.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PermutationStatisticSummary {
    /// Number of permutation losses observed.
    pub count: u64,
    /// Online mean loss.
    pub mean: f64,
    /// Online sample standard deviation (denominator `count - 1`).
    pub sample_std_dev: f64,
    /// Minimum observed loss.
    pub min: f64,
    /// Maximum observed loss.
    pub max: f64,
    /// Approximate 25th percentile (P² estimator).
    pub approximate_p25: f64,
    /// Approximate 50th percentile (P² estimator).
    pub approximate_p50: f64,
    /// Approximate 75th percentile (P² estimator).
    pub approximate_p75: f64,
}

/// Result of [`robinson_turner_two_sample_test`].
///
/// `extreme_count` is the exact extreme-labeling count `b` in exact mode and
/// `b_random + 1` in Monte-Carlo mode. In exact mode `random_seed` is `None`
/// because no randomness is used. `distance_matrix` is `None` unless
/// `return_distance_matrix` was set; when serialized, it stays `null`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RobinsonTurnerTestResult {
    /// Observed within-group loss `F_{p,q}` for the supplied labeling.
    pub statistic: f64,
    /// Lower-tail permutation p-value (`b / C(N,n1)` or `(b_random+1)/(B+1)`).
    pub p_value: f64,
    /// `p_value < alpha`.
    pub reject_null: bool,
    /// Significance level used.
    pub alpha: f64,
    /// Requested homology degree.
    pub homology_dim: usize,
    /// Diagram-distance order used.
    pub diagram_distance: DiagramDistance,
    /// Ground metric of the diagram distance; always `"L-infinity"`.
    pub ground_metric: String,
    /// Loss exponent `q`.
    pub loss_q: u32,
    /// `(n1, n2)`.
    pub group_sizes: (usize, usize),
    /// Inference mode actually used.
    pub inference_mode: InferenceMode,
    /// `C(N, n1)`, the total number of allocations preserving group sizes.
    pub n_labelings_total: u128,
    /// `B` in Monte-Carlo mode; `0` in exact mode.
    pub n_random_permutations: u64,
    /// Exact `b`, or `b_random + 1` in Monte-Carlo mode.
    pub extreme_count: u64,
    /// Minimum nonzero p-value increment (`1/C(N,n1)` or `1/(B+1)`).
    pub p_value_resolution: f64,
    /// Seed used (supplied or `OsRng`-generated); `None` in exact mode.
    pub random_seed: Option<u64>,
    /// Persistent-homology / diagram-distance backend name.
    pub backend: String,
    /// Pinned backend version string.
    pub backend_version: String,
    /// Filtration provenance.
    pub ph_configuration: PhConfiguration,
    /// Finite pair counts of the pooled diagrams (group A then group B).
    pub diagram_sizes: Vec<usize>,
    /// Number of essential degree-`k` classes dropped per pooled cloud
    /// (all zeros under `Reject`).
    pub dropped_essential_classes: Vec<usize>,
    /// Online summary over permutation losses (excluding the observed
    /// allocation in Monte-Carlo mode; over all allocations in exact mode).
    pub permutation_summary: PermutationStatisticSummary,
    /// Non-fatal issues (e.g. resolution coarser than `alpha`, dropped
    /// essential classes).
    pub warnings: Vec<String>,
    /// Mechanically derived interpretation sentence.
    pub interpretation: String,
    /// Symmetric `N x N` diagram-distance matrix, if requested.
    pub distance_matrix: Option<Vec<Vec<f64>>>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Error type for [`robinson_turner_two_sample_test`] and
/// [`PointCloud::try_from_rows`]. No variant is produced by a panic.
#[derive(Debug, Error)]
pub enum RobinsonTurnerError {
    /// A group had fewer than two clouds.
    #[error("group {group} must contain at least 2 point clouds, got {got}")]
    InvalidGroupSize {
        /// Which group ("A" or "B").
        group: &'static str,
        /// Number of clouds supplied.
        got: usize,
    },
    /// A cloud was empty, ragged, or zero-dimensional.
    #[error("point cloud {index} is malformed: {reason}")]
    MalformedCloud {
        /// Cloud index within its group (`usize::MAX` for a whole-group issue).
        index: usize,
        /// Human-readable reason.
        reason: String,
    },
    /// Ambient dimensions differ across clouds.
    #[error("inconsistent ambient dimension: expected {expected}, cloud {index} has {got}")]
    InconsistentDimension {
        /// Common ambient dimension.
        expected: usize,
        /// Pooled cloud index.
        index: usize,
        /// Dimension found.
        got: usize,
    },
    /// A coordinate was NaN or infinite.
    #[error("point cloud {index} contains a non-finite coordinate at offset {offset}")]
    NonFiniteCoordinate {
        /// Cloud index.
        index: usize,
        /// Coordinate offset within the cloud.
        offset: usize,
    },
    /// Invalid configuration field.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// `C(N, n1)` overflowed `u128`.
    #[error("label space size C({n}, {n1}) exceeds the u128 range")]
    LabelSpaceOverflow {
        /// Pooled cloud count.
        n: usize,
        /// Group-A size.
        n1: usize,
    },
    /// A requested-degree class never died within the truncated filtration and
    /// the policy is [`EssentialClassPolicy::Reject`].
    #[error(
        "cloud {index} has {count} essential H{dim} class(es) that never die within the filtration; use EssentialClassPolicy::Drop to censor them"
    )]
    EssentialClasses {
        /// Pooled cloud index.
        index: usize,
        /// Homology degree.
        dim: usize,
        /// Number of essential classes.
        count: usize,
    },
    /// The `oxicuda-tda` backend failed.
    #[error("oxicuda-tda backend failure: {0}")]
    Backend(#[source] Box<dyn std::error::Error + Send + Sync>),
}

fn backend_err(e: impl std::error::Error + Send + Sync + 'static) -> RobinsonTurnerError {
    RobinsonTurnerError::Backend(Box::new(e))
}

#[derive(Debug)]
struct StringError(String);
impl std::fmt::Display for StringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for StringError {}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_cloud(cloud: &PointCloud, index: usize) -> Result<(), RobinsonTurnerError> {
    if cloud.ambient_dim == 0 {
        return Err(RobinsonTurnerError::MalformedCloud {
            index,
            reason: "ambient dimension is zero".to_string(),
        });
    }
    if cloud.n_points == 0 {
        return Err(RobinsonTurnerError::MalformedCloud {
            index,
            reason: "cloud has no points".to_string(),
        });
    }
    let expected_len = match cloud.n_points.checked_mul(cloud.ambient_dim) {
        Some(len) => len,
        None => {
            return Err(RobinsonTurnerError::MalformedCloud {
                index,
                reason: "n_points * ambient_dim overflows usize".to_string(),
            });
        }
    };
    if cloud.coordinates.len() != expected_len {
        return Err(RobinsonTurnerError::MalformedCloud {
            index,
            reason: format!(
                "coordinates length {} != n_points * ambient_dim = {}",
                cloud.coordinates.len(),
                expected_len
            ),
        });
    }
    for (offset, &x) in cloud.coordinates.iter().enumerate() {
        if !x.is_finite() {
            return Err(RobinsonTurnerError::NonFiniteCoordinate { index, offset });
        }
    }
    Ok(())
}

/// `C(n, k)` with checked `u128` arithmetic; `None` on overflow.
fn checked_binomial(n: usize, k: usize) -> Option<u128> {
    let k = k.min(n.checked_sub(k)?);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc.checked_mul((n - k + i + 1) as u128)?;
        acc /= (i + 1) as u128;
    }
    Some(acc)
}

// ---------------------------------------------------------------------------
// Persistent-homology adapter (oxicuda-tda kept fully internal)
// ---------------------------------------------------------------------------

/// Map one cloud to its degree-`k` diagram, applying the essential-class
/// policy. Returns the finite-pair diagram and the number of degree-`k`
/// essential classes that were dropped.
fn cloud_to_diagram(
    cloud: &PointCloud,
    index: usize,
    config: &RobinsonTurnerConfig,
) -> Result<(PersistenceDiagram, usize), RobinsonTurnerError> {
    // oxicuda-tda 0.5.5 panics inside `vietoris_rips` when `max_dim + 1`
    // exceeds the point count (it seeds subset enumeration with `(0..size)`
    // unconditionally). Clamp to the largest possible simplex dimension
    // `n_points - 1`; classes of degree `k` that cannot be killed within a
    // clamped filtration simply surface as essential, which the policy below
    // already handles.
    let max_simplex_dim = (config.homology_dim + 1).min(cloud.n_points.saturating_sub(1));
    let filtration = Filtration::vietoris_rips_from_points(
        &cloud.coordinates,
        cloud.ambient_dim,
        config.max_edge_length,
        max_simplex_dim,
    )
    .map_err(backend_err)?;
    let mut boundary = BoundaryMatrix::from_filtration(&filtration).map_err(backend_err)?;
    reduce_boundary_matrix(&mut boundary);
    let pairs = extract_persistence_pairs(&boundary, &filtration).map_err(backend_err)?;

    let mut finite: Vec<PersistencePair> = Vec::new();
    let mut n_essential = 0usize;
    for p in pairs {
        if p.dim != config.homology_dim {
            continue;
        }
        if p.is_essential() {
            n_essential += 1;
        } else {
            finite.push(p);
        }
    }
    if n_essential > 0 && config.essential_class_policy == EssentialClassPolicy::Reject {
        return Err(RobinsonTurnerError::EssentialClasses {
            index,
            dim: config.homology_dim,
            count: n_essential,
        });
    }
    Ok((
        PersistenceDiagram::new(finite, config.homology_dim),
        n_essential,
    ))
}

/// Dispatch to the optimal pinned-backend diagram distance. Each invocation is
/// counted once in the doc-hidden [`__test_seams`] counter.
fn diagram_distance(
    a: &PersistenceDiagram,
    b: &PersistenceDiagram,
    kind: DiagramDistance,
) -> Result<f64, RobinsonTurnerError> {
    PAIRWISE_DISTANCE_CALLS.with(|c| c.set(c.get() + 1));
    let d = match kind {
        DiagramDistance::Wasserstein1 => wasserstein_p(a, b, 1.0).map_err(backend_err)?,
        DiagramDistance::Wasserstein2 => wasserstein_p(a, b, 2.0).map_err(backend_err)?,
        DiagramDistance::Bottleneck => bottleneck_distance(a, b).map_err(backend_err)?,
    };
    if !d.is_finite() {
        return Err(backend_err(StringError(
            "backend returned a non-finite diagram distance".to_string(),
        )));
    }
    Ok(d)
}

/// Build the symmetric `N x N` distance matrix once: upper triangle computed,
/// mirrored, exact zero diagonal. Serial by design.
fn distance_matrix(
    diagrams: &[PersistenceDiagram],
    kind: DiagramDistance,
) -> Result<Vec<Vec<f64>>, RobinsonTurnerError> {
    let n = diagrams.len();
    let mut m = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = diagram_distance(&diagrams[i], &diagrams[j], kind)?;
            m[i][j] = d;
            m[j][i] = d;
        }
    }
    Ok(m)
}

// ---------------------------------------------------------------------------
// Joint-loss statistic
// ---------------------------------------------------------------------------

/// `sum_{i<j in idx} d(i,j)^q / (n(n-1))` — one group's contribution.
fn group_loss(mat: &[Vec<f64>], idx: &[usize], q: u32) -> f64 {
    let n = idx.len();
    if n < 2 {
        return 0.0;
    }
    let mut s = 0.0;
    for a in 0..n {
        for b in (a + 1)..n {
            let d = mat[idx[a]][idx[b]];
            s += if q == 2 { d * d } else { d };
        }
    }
    s / (n * (n - 1)) as f64
}

/// `F_{p,q}` for a labeling given as the group-A index set.
fn allocation_loss(mat: &[Vec<f64>], idx_a: &[usize], idx_b: &[usize], q: u32) -> f64 {
    group_loss(mat, idx_a, q) + group_loss(mat, idx_b, q)
}

fn complement(n: usize, idx_a: &[usize]) -> Vec<usize> {
    let mut in_a = vec![false; n];
    for &i in idx_a {
        in_a[i] = true;
    }
    (0..n).filter(|&i| !in_a[i]).collect()
}

// ---------------------------------------------------------------------------
// Online permutation-loss summary (Welford + min/max + P² quartiles)
// ---------------------------------------------------------------------------

/// Deterministic 5-marker P² estimator for quartiles (Jain–Chlamtac, p = 0.5
/// formulation tracks min/p25/p50/p75/max).
struct P2Quartiles {
    head: Vec<f64>,
    heights: [f64; 5],
    positions: [i64; 5],
    desired: [f64; 5],
}

const P2_INCREMENTS: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

impl P2Quartiles {
    fn new() -> Self {
        Self {
            head: Vec::with_capacity(5),
            heights: [0.0; 5],
            positions: [0; 5],
            desired: [0.0; 5],
        }
    }

    fn push(&mut self, x: f64) {
        if self.head.len() < 5 {
            self.head.push(x);
            if self.head.len() == 5 {
                self.head.sort_by(f64::total_cmp);
                self.heights = self.head.clone().try_into().unwrap_or([0.0; 5]);
                self.positions = [1, 2, 3, 4, 5];
                self.desired = [1.0, 2.0, 3.0, 4.0, 5.0];
            }
            return;
        }
        let q = &mut self.heights;
        let n = &mut self.positions;
        let np = &mut self.desired;
        // Locate the cell containing x.
        let k = if x < q[0] {
            q[0] = x;
            0
        } else if x >= q[4] {
            q[4] = x;
            3
        } else {
            (0..4).find(|&i| q[i] <= x && x < q[i + 1]).unwrap_or(3)
        };
        for ni in n.iter_mut().skip(k + 1) {
            *ni += 1;
        }
        for i in 0..5 {
            np[i] += P2_INCREMENTS[i];
        }
        for i in 1..4 {
            let d = np[i] - n[i] as f64;
            if (d >= 1.0 && n[i + 1] - n[i] > 1) || (d <= -1.0 && n[i - 1] - n[i] < -1) {
                let dir = d.signum();
                let di = dir as i64;
                let ii = i as i64;
                // Parabolic prediction.
                let qp = q[i]
                    + dir / (n[i + 1] - n[i - 1]) as f64
                        * ((n[i] - n[i - 1] + di) as f64 * (q[i + 1] - q[i])
                            / (n[i + 1] - n[i]) as f64
                            + (n[i + 1] - n[i] - di) as f64 * (q[i] - q[i - 1])
                                / (n[i] - n[i - 1]) as f64);
                let ok = if dir > 0.0 {
                    q[i] < qp && qp < q[i + 1]
                } else {
                    q[i - 1] < qp && qp < q[i]
                };
                q[i] = if ok {
                    qp
                } else {
                    // Linear fallback.
                    q[i] + dir * (q[(ii + di) as usize] - q[i])
                        / (n[(ii + di) as usize] - n[i]) as f64
                };
                n[i] += di;
            }
        }
    }

    /// `(p25, p50, p75)`; for fewer than 5 observations use exact type-7
    /// quantiles of the retained head buffer.
    fn quartiles(&self) -> (f64, f64, f64) {
        if self.head.len() < 5 {
            let mut v = self.head.clone();
            v.sort_by(f64::total_cmp);
            let q = |p: f64| -> f64 {
                if v.is_empty() {
                    return f64::NAN;
                }
                if v.len() == 1 {
                    return v[0];
                }
                let h = p * (v.len() - 1) as f64;
                let lo = h.floor() as usize;
                let hi = (lo + 1).min(v.len() - 1);
                v[lo] + (h - lo as f64) * (v[hi] - v[lo])
            };
            (q(0.25), q(0.5), q(0.75))
        } else {
            (self.heights[1], self.heights[2], self.heights[3])
        }
    }
}

/// Online loss summary: Welford mean/M2, min, max, P² quartiles.
struct OnlineSummary {
    count: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    p2: P2Quartiles,
}

impl OnlineSummary {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            p2: P2Quartiles::new(),
        }
    }

    fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
        self.p2.push(x);
    }

    fn finish(&self) -> PermutationStatisticSummary {
        let (p25, p50, p75) = self.p2.quartiles();
        PermutationStatisticSummary {
            count: self.count,
            mean: if self.count > 0 { self.mean } else { f64::NAN },
            sample_std_dev: if self.count > 1 {
                (self.m2 / (self.count - 1) as f64).sqrt()
            } else {
                f64::NAN
            },
            min: if self.count > 0 { self.min } else { f64::NAN },
            max: if self.count > 0 { self.max } else { f64::NAN },
            approximate_p25: p25,
            approximate_p50: p50,
            approximate_p75: p75,
        }
    }
}

// ---------------------------------------------------------------------------
// Permutation scans
// ---------------------------------------------------------------------------

/// Parameters shared by the permutation scans.
struct ScanParams<'a> {
    mat: &'a [Vec<f64>],
    n: usize,
    n1: usize,
    q: u32,
    observed: f64,
    tol: f64,
}

/// Lazily enumerate all `C(N, n1)` allocations (including the observed one).
/// Returns `(extreme_count, visited_count, summary)`.
fn exact_scan(params: &ScanParams<'_>) -> (u64, u128, OnlineSummary) {
    let mut summary = OnlineSummary::new();
    let mut extreme = 0u64;
    let mut visited = 0u128;
    for combo in (0..params.n).combinations(params.n1) {
        let idx_b = complement(params.n, &combo);
        let f = allocation_loss(params.mat, &combo, &idx_b, params.q);
        summary.push(f);
        visited += 1;
        if f <= params.observed + params.tol {
            extreme += 1;
        }
    }
    (extreme, visited, summary)
}

/// Draw `b` uniform allocations; each draw shuffles a fresh `0..N` index vector
/// and takes its first `n1` entries. Returns `(b_random, summary)`.
fn mc_scan(params: &ScanParams<'_>, b: u64, seed: u64) -> (u64, OnlineSummary) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut summary = OnlineSummary::new();
    let mut b_random = 0u64;
    for _ in 0..b {
        let mut idx: Vec<usize> = (0..params.n).collect();
        idx.shuffle(&mut rng);
        let (idx_a, idx_b) = idx.split_at(params.n1);
        let f = allocation_loss(params.mat, idx_a, idx_b, params.q);
        summary.push(f);
        if f <= params.observed + params.tol {
            b_random += 1;
        }
    }
    (b_random, summary)
}

// ---------------------------------------------------------------------------
// Test seam (public but doc-hidden) for the integration-test shard
// ---------------------------------------------------------------------------

// Thread-local because Rust tests execute in parallel threads within one
// process: a global counter would race across tests. The engine is fully
// serial, so each call to `robinson_turner_two_sample_test` stays on the
// calling thread and the per-thread count is exact.
thread_local! {
    static PAIRWISE_DISTANCE_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[doc(hidden)]
pub mod __test_seams {
    use super::PAIRWISE_DISTANCE_CALLS;

    /// Reset the calling thread's pairwise diagram-distance invocation
    /// counter (serial engine).
    pub fn reset_pairwise_distance_calls() {
        PAIRWISE_DISTANCE_CALLS.with(|c| c.set(0));
    }

    /// Number of times the backend pairwise distance function was invoked on
    /// the calling thread since the last reset.
    pub fn pairwise_distance_calls() -> usize {
        PAIRWISE_DISTANCE_CALLS.with(|c| c.get())
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Robinson–Turner two-sample randomization test on degree-`k` persistence
/// diagrams of two independent groups of point clouds.
///
/// Each cloud independently yields exactly one degree-`config.homology_dim`
/// diagram (Euclidean Vietoris–Rips through simplex dimension `k + 1`,
/// truncated at `config.max_edge_length`, coefficients in `Z/2`). The
/// `N x N` optimal diagram-distance matrix is built once and reused for every
/// labeling; permutations act at cloud level only.
///
/// The p-value is lower-tailed: a meaningful supplied grouping is expected to
/// have unusually *small* within-group dispersion. Exact mode enumerates all
/// `C(N, n1)` allocations including the observed one; Monte-Carlo mode draws
/// `B = config.n_permutations` fresh shuffles and returns the corrected
/// `(b_random + 1) / (B + 1)` (never zero; default `B = 9999` → resolution
/// `0.0001`). The used `u64` seed (supplied or `OsRng`-generated) is recorded
/// for reproducibility.
pub fn robinson_turner_two_sample_test(
    group_a: &[PointCloud],
    group_b: &[PointCloud],
    config: &RobinsonTurnerConfig,
) -> Result<RobinsonTurnerTestResult, RobinsonTurnerError> {
    // --- Validation -------------------------------------------------------
    if group_a.len() < 2 {
        return Err(RobinsonTurnerError::InvalidGroupSize {
            group: "A",
            got: group_a.len(),
        });
    }
    if group_b.len() < 2 {
        return Err(RobinsonTurnerError::InvalidGroupSize {
            group: "B",
            got: group_b.len(),
        });
    }
    let pooled: Vec<&PointCloud> = group_a.iter().chain(group_b.iter()).collect();
    let n = pooled.len();
    let n1 = group_a.len();
    for (i, c) in pooled.iter().enumerate() {
        validate_cloud(c, i)?;
    }
    let ambient_dim = pooled[0].ambient_dim;
    for (i, c) in pooled.iter().enumerate() {
        if c.ambient_dim != ambient_dim {
            return Err(RobinsonTurnerError::InconsistentDimension {
                expected: ambient_dim,
                index: i,
                got: c.ambient_dim,
            });
        }
    }
    if config.homology_dim > MAX_HOMOLOGY_DIM {
        return Err(RobinsonTurnerError::InvalidConfig(format!(
            "homology_dim {} exceeds backend limit {MAX_HOMOLOGY_DIM} (filtration dim <= 6)",
            config.homology_dim
        )));
    }
    if !config.max_edge_length.is_finite() || config.max_edge_length <= 0.0 {
        return Err(RobinsonTurnerError::InvalidConfig(format!(
            "max_edge_length must be finite and strictly positive, got {}",
            config.max_edge_length
        )));
    }
    if config.loss_q != 1 && config.loss_q != 2 {
        return Err(RobinsonTurnerError::InvalidConfig(format!(
            "loss_q must be 1 or 2, got {}",
            config.loss_q
        )));
    }
    if !config.alpha.is_finite() || config.alpha <= 0.0 || config.alpha >= 1.0 {
        return Err(RobinsonTurnerError::InvalidConfig(format!(
            "alpha must be in (0, 1), got {}",
            config.alpha
        )));
    }
    if !config.tie_tolerance.is_finite() || config.tie_tolerance < 0.0 {
        return Err(RobinsonTurnerError::InvalidConfig(format!(
            "tie_tolerance must be finite and nonnegative, got {}",
            config.tie_tolerance
        )));
    }
    if config.method == InferenceMethod::MonteCarlo && config.n_permutations == 0 {
        return Err(RobinsonTurnerError::InvalidConfig(
            "n_permutations (B) must be nonzero for Monte-Carlo inference".to_string(),
        ));
    }

    let n_labelings =
        checked_binomial(n, n1).ok_or(RobinsonTurnerError::LabelSpaceOverflow { n, n1 })?;

    // --- Diagrams ----------------------------------------------------------
    let mut diagrams = Vec::with_capacity(n);
    let mut dropped = Vec::with_capacity(n);
    for (i, c) in pooled.iter().enumerate() {
        let (d, n_essential) = cloud_to_diagram(c, i, config)?;
        diagrams.push(d);
        dropped.push(n_essential);
    }
    let diagram_sizes: Vec<usize> = diagrams.iter().map(|d| d.pairs.len()).collect();

    // --- Distance matrix (computed once) -----------------------------------
    let mat = distance_matrix(&diagrams, config.diagram_distance)?;

    // --- Observed statistic -------------------------------------------------
    let idx_a: Vec<usize> = (0..n1).collect();
    let idx_b: Vec<usize> = (n1..n).collect();
    let observed = allocation_loss(&mat, &idx_a, &idx_b, config.loss_q);
    let tol = config.tie_tolerance;

    // --- Inference ----------------------------------------------------------
    let use_exact = match config.method {
        InferenceMethod::Exact => {
            if n_labelings > config.max_exact_labelings {
                return Err(RobinsonTurnerError::InvalidConfig(format!(
                    "exact inference requires enumerating {n_labelings} labelings, exceeding \
                     max_exact_labelings = {}",
                    config.max_exact_labelings
                )));
            }
            true
        }
        InferenceMethod::Auto => n_labelings <= config.max_exact_labelings,
        InferenceMethod::MonteCarlo => false,
    };

    let mut warnings = Vec::new();
    let (mode, p_value, extreme_count, resolution, random_seed, n_random, summary) = if use_exact {
        let params = ScanParams {
            mat: &mat,
            n,
            n1,
            q: config.loss_q,
            observed,
            tol,
        };
        let (b, _visited, summary) = exact_scan(&params);
        (
            InferenceMode::Exact,
            b as f64 / n_labelings as f64,
            b,
            1.0 / n_labelings as f64,
            None,
            0u64,
            summary.finish(),
        )
    } else {
        if config.n_permutations == 0 {
            return Err(RobinsonTurnerError::InvalidConfig(
                "n_permutations (B) must be nonzero for Monte-Carlo inference".to_string(),
            ));
        }
        let b_draws = config.n_permutations;
        let seed = match config.random_seed {
            Some(s) => s,
            None => {
                // rand 0.9: OsRng implements `TryRngCore` (its OS call can
                // fail); propagate instead of panicking.
                use rand::rand_core::TryRngCore;
                OsRng.try_next_u64().map_err(|e| {
                    RobinsonTurnerError::InvalidConfig(format!(
                        "failed to draw a random seed from OsRng: {e}"
                    ))
                })?
            }
        };
        let params = ScanParams {
            mat: &mat,
            n,
            n1,
            q: config.loss_q,
            observed,
            tol,
        };
        let (b_random, summary) = mc_scan(&params, b_draws, seed);
        let p = (b_random + 1) as f64 / (b_draws + 1) as f64;
        (
            InferenceMode::MonteCarlo,
            p,
            b_random + 1,
            1.0 / (b_draws + 1) as f64,
            Some(seed),
            b_draws,
            summary.finish(),
        )
    };

    if resolution > config.alpha {
        warnings.push(format!(
            "p-value resolution {resolution} exceeds alpha = {}; the attainable p-value grid \
             is coarser than the significance level",
            config.alpha
        ));
    }
    let total_dropped: usize = dropped.iter().sum();
    if total_dropped > 0 {
        warnings.push(format!(
            "{total_dropped} essential H{} class(es) were dropped under EssentialClassPolicy::Drop \
             (censoring; the filtration ended before they died)",
            config.homology_dim
        ));
    }

    let reject_null = p_value < config.alpha;
    let interpretation = format!(
        "{} The statistic concerns group differences in the selected degree-{} \
         persistent-homology summary, not feature significance for an individual cloud.",
        if reject_null {
            format!(
                "Reject label exchangeability at alpha = {}; the groups have evidence of \
                 different {}-dimensional persistence-diagram distributions under this analysis \
                 pipeline.",
                config.alpha, config.homology_dim
            )
        } else {
            format!(
                "Do not reject label exchangeability at alpha = {}; this is not evidence that \
                 the distributions are equal.",
                config.alpha
            )
        },
        config.homology_dim
    );

    Ok(RobinsonTurnerTestResult {
        statistic: observed,
        p_value,
        reject_null,
        alpha: config.alpha,
        homology_dim: config.homology_dim,
        diagram_distance: config.diagram_distance,
        ground_metric: "L-infinity".to_string(),
        loss_q: config.loss_q,
        group_sizes: (n1, group_b.len()),
        inference_mode: mode,
        n_labelings_total: n_labelings,
        n_random_permutations: n_random,
        extreme_count,
        p_value_resolution: resolution,
        random_seed,
        backend: BACKEND_NAME.to_string(),
        backend_version: BACKEND_VERSION.to_string(),
        ph_configuration: PhConfiguration {
            homology_dim: config.homology_dim,
            max_edge_length: config.max_edge_length,
            max_simplex_dim: config.homology_dim + 1,
            coefficient_field: "Z/2".to_string(),
            filtration: "Vietoris-Rips (Euclidean)".to_string(),
            essential_class_policy: config.essential_class_policy,
        },
        diagram_sizes,
        dropped_essential_classes: dropped,
        permutation_summary: summary,
        warnings,
        interpretation,
        distance_matrix: if config.return_distance_matrix {
            Some(mat)
        } else {
            None
        },
    })
}

// ---------------------------------------------------------------------------
// Unit tests (plan items 1-10; serde_json item 11 belongs to the tests shard)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a symmetric matrix from scalar positions: d[i][j] = |x_i - x_j|.
    fn scalar_matrix(xs: &[f64]) -> Vec<Vec<f64>> {
        let n = xs.len();
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                m[i][j] = (xs[i] - xs[j]).abs();
            }
        }
        m
    }

    fn sample_variance(xs: &[f64]) -> f64 {
        let n = xs.len();
        let mean = xs.iter().sum::<f64>() / n as f64;
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    }

    fn tiny_cloud(rows: &[&[f64]]) -> PointCloud {
        PointCloud::try_from_rows(rows.iter().map(|r| r.to_vec()).collect()).unwrap()
    }

    // 1. Pairwise-loss identity: F_{2,2} equals the sum of the groups'
    //    unbiased sample variances for scalar distances d = |x_i - x_j|.
    #[test]
    fn loss_equals_sum_of_sample_variances() {
        let xs = [0.0, 1.0, 2.0, 5.0, 9.0, 14.0];
        let mat = scalar_matrix(&xs);
        // Try several partitions.
        for (a, b) in [
            (vec![0, 1, 2], vec![3, 4, 5]),
            (vec![0, 3], vec![1, 2, 4, 5]),
            (vec![1, 4], vec![0, 2, 3, 5]),
        ] {
            let f = allocation_loss(&mat, &a, &b, 2);
            let va = sample_variance(&a.iter().map(|&i| xs[i]).collect::<Vec<_>>());
            let vb = sample_variance(&b.iter().map(|&i| xs[i]).collect::<Vec<_>>());
            assert!((f - (va + vb)).abs() < 1e-12, "F={f} vs {va}+{vb}");
        }
    }

    // 2. Hand-computed statistic on a fixed 4x4 matrix, two groups of two.
    #[test]
    fn hand_computed_statistic() {
        // d(0,1)=2, d(2,3)=4, others irrelevant for F when groups are {0,1},{2,3}.
        let mut mat = vec![vec![0.0; 4]; 4];
        mat[0][1] = 2.0;
        mat[1][0] = 2.0;
        mat[2][3] = 4.0;
        mat[3][2] = 4.0;
        // F_{p,2} = 2^2/(2*1) + 4^2/(2*1) = 2 + 8 = 10.
        assert_eq!(allocation_loss(&mat, &[0, 1], &[2, 3], 2), 10.0);
        // q = 1: 2/2 + 4/2 = 3.
        assert_eq!(allocation_loss(&mat, &[0, 1], &[2, 3], 1), 3.0);
    }

    // 3. All-zero matrix: every labeling ties; exact p-value is 1.0.
    #[test]
    fn all_zero_ties() {
        let mat = vec![vec![0.0; 4]; 4];
        let params = ScanParams {
            mat: &mat,
            n: 4,
            n1: 2,
            q: 2,
            observed: 0.0,
            tol: 0.0,
        };
        let (b, visited, _s) = exact_scan(&params);
        assert_eq!(visited, 6);
        assert_eq!(b, 6);
        assert_eq!(b as f64 / visited as f64, 1.0);
    }

    // 4. Exact enumeration visits all C(4,2) = 6 allocations including the
    //    observed allocation {0,1}.
    #[test]
    fn exact_enumeration_visits_all_allocations() {
        let mat = scalar_matrix(&[0.0, 0.1, 5.0, 5.1]);
        let observed = allocation_loss(&mat, &[0, 1], &[2, 3], 2);
        let mut seen_observed = false;
        let mut count = 0usize;
        for combo in (0..4).combinations(2) {
            count += 1;
            if combo == vec![0, 1] {
                seen_observed = true;
            }
            let idx_b = complement(4, &combo);
            assert_eq!(combo.len() + idx_b.len(), 4);
            let _ = allocation_loss(&mat, &combo, &idx_b, 2);
        }
        assert_eq!(count, 6);
        assert!(seen_observed);
        // Observed is the unique-ish minimum here.
        let params = ScanParams {
            mat: &mat,
            n: 4,
            n1: 2,
            q: 2,
            observed,
            tol: 0.0,
        };
        let (b, _, _) = exact_scan(&params);
        assert!(b >= 1);
    }

    // 5. Monte-Carlo correction: b_random extreme draws give (b_random+1)/(B+1),
    //    never zero; consistent with extreme_count field.
    #[test]
    fn monte_carlo_correction() {
        // Matrix where observed allocation is the unique minimum together with
        // its swapped complement: with the fixed seed no extreme draw occurs,
        // so p = 1/(B+1).
        let mut mat = vec![vec![1.0; 6]; 6];
        for (i, row) in mat.iter_mut().enumerate() {
            row[i] = 0.0;
        }
        mat[0][1] = 0.0;
        mat[1][0] = 0.0;
        mat[0][2] = 0.0;
        mat[2][0] = 0.0;
        mat[1][2] = 0.0;
        mat[2][1] = 0.0;
        mat[3][4] = 0.0;
        mat[4][3] = 0.0;
        mat[3][5] = 0.0;
        mat[5][3] = 0.0;
        mat[4][5] = 0.0;
        mat[5][4] = 0.0;
        let observed = allocation_loss(&mat, &[0, 1, 2], &[3, 4, 5], 2);
        assert_eq!(observed, 0.0);
        let b = 50u64;
        let params = ScanParams {
            mat: &mat,
            n: 6,
            n1: 3,
            q: 2,
            observed,
            tol: 0.0,
        };
        let (b_random, _s) = mc_scan(&params, b, 0xC0FFEE);
        // Only allocations {0,1,2}|{3,4,5} or its swap tie; whatever b_random
        // is, the correction is (b_random+1)/(B+1) and nonzero.
        let p = (b_random + 1) as f64 / (b + 1) as f64;
        assert!(p > 0.0);
        assert!(p >= 1.0 / (b + 1) as f64);
        // Deterministic all-extreme case: every permutation ties an all-equal
        // matrix, so p = (B+1)/(B+1) = 1.
        let zero = vec![vec![0.0; 6]; 6];
        let zparams = ScanParams {
            mat: &zero,
            n: 6,
            n1: 3,
            q: 2,
            observed: 0.0,
            tol: 0.0,
        };
        let (b_all, _) = mc_scan(&zparams, b, 7);
        assert_eq!(b_all, b);
        assert_eq!((b_all + 1) as f64 / (b + 1) as f64, 1.0);
    }

    // 6. Seed reproducibility through the public API.
    #[test]
    fn seed_reproducibility() {
        let clouds_a = [
            tiny_cloud(&[&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]]),
            tiny_cloud(&[&[0.0, 0.0], &[1.1, 0.0], &[0.0, 1.1]]),
            tiny_cloud(&[&[0.2, 0.0], &[1.0, 0.2], &[0.0, 1.0]]),
        ];
        let clouds_b = [
            tiny_cloud(&[&[0.0, 0.0], &[3.0, 0.0], &[0.0, 3.0]]),
            tiny_cloud(&[&[0.0, 0.0], &[2.9, 0.1], &[0.1, 3.0]]),
            tiny_cloud(&[&[0.0, 0.1], &[3.1, 0.0], &[0.0, 2.9]]),
        ];
        let mut cfg = RobinsonTurnerConfig::new(1, 4.0);
        cfg.method = InferenceMethod::MonteCarlo;
        cfg.n_permutations = 200;
        cfg.random_seed = Some(42);
        cfg.essential_class_policy = EssentialClassPolicy::Drop;
        let r1 = robinson_turner_two_sample_test(&clouds_a, &clouds_b, &cfg).unwrap();
        let r2 = robinson_turner_two_sample_test(&clouds_a, &clouds_b, &cfg).unwrap();
        assert_eq!(r1, r2);
        assert_eq!(r1.random_seed, Some(42));
        assert_eq!(r1.inference_mode, InferenceMode::MonteCarlo);
    }

    // 7. Input validation matrix.
    #[test]
    fn validation_matrix() {
        let c = || tiny_cloud(&[&[0.0, 0.0], &[1.0, 0.0]]);
        let good = RobinsonTurnerConfig::new(0, 2.0);
        // Small groups.
        assert!(matches!(
            robinson_turner_two_sample_test(&[c()], &[c(), c()], &good),
            Err(RobinsonTurnerError::InvalidGroupSize { group: "A", .. })
        ));
        // Ragged cloud.
        let ragged = PointCloud {
            coordinates: vec![0.0, 0.0, 1.0],
            n_points: 2,
            ambient_dim: 2,
        };
        assert!(matches!(
            robinson_turner_two_sample_test(&[ragged, c()], &[c(), c()], &good),
            Err(RobinsonTurnerError::MalformedCloud { .. })
        ));
        // Empty cloud.
        let empty = PointCloud {
            coordinates: vec![],
            n_points: 0,
            ambient_dim: 2,
        };
        assert!(matches!(
            robinson_turner_two_sample_test(&[empty, c()], &[c(), c()], &good),
            Err(RobinsonTurnerError::MalformedCloud { .. })
        ));
        // Inconsistent ambient dim.
        let d3 = tiny_cloud(&[&[0.0, 0.0, 0.0]]);
        assert!(matches!(
            robinson_turner_two_sample_test(&[d3, c()], &[c(), c()], &good),
            Err(RobinsonTurnerError::InconsistentDimension { .. })
        ));
        // NaN / infinity.
        let nan = PointCloud {
            coordinates: vec![0.0, f64::NAN],
            n_points: 1,
            ambient_dim: 2,
        };
        assert!(matches!(
            robinson_turner_two_sample_test(&[nan, c()], &[c(), c()], &good),
            Err(RobinsonTurnerError::NonFiniteCoordinate { .. })
        ));
        let inf = PointCloud {
            coordinates: vec![0.0, f64::INFINITY],
            n_points: 1,
            ambient_dim: 2,
        };
        assert!(matches!(
            robinson_turner_two_sample_test(&[inf, c()], &[c(), c()], &good),
            Err(RobinsonTurnerError::NonFiniteCoordinate { .. })
        ));
        // Invalid homology_dim.
        let mut bad = good.clone();
        bad.homology_dim = 6;
        assert!(matches!(
            robinson_turner_two_sample_test(&[c(), c()], &[c(), c()], &bad),
            Err(RobinsonTurnerError::InvalidConfig(_))
        ));
        // Edge length: zero, negative, NaN, infinite.
        for v in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut bad = good.clone();
            bad.max_edge_length = v;
            assert!(matches!(
                robinson_turner_two_sample_test(&[c(), c()], &[c(), c()], &bad),
                Err(RobinsonTurnerError::InvalidConfig(_))
            ));
        }
        // q.
        let mut bad = good.clone();
        bad.loss_q = 3;
        assert!(matches!(
            robinson_turner_two_sample_test(&[c(), c()], &[c(), c()], &bad),
            Err(RobinsonTurnerError::InvalidConfig(_))
        ));
        // alpha.
        for v in [0.0, 1.0, -0.5, f64::NAN] {
            let mut bad = good.clone();
            bad.alpha = v;
            assert!(matches!(
                robinson_turner_two_sample_test(&[c(), c()], &[c(), c()], &bad),
                Err(RobinsonTurnerError::InvalidConfig(_))
            ));
        }
        // tolerance.
        let mut bad = good.clone();
        bad.tie_tolerance = -1.0;
        assert!(matches!(
            robinson_turner_two_sample_test(&[c(), c()], &[c(), c()], &bad),
            Err(RobinsonTurnerError::InvalidConfig(_))
        ));
        // B = 0 with MonteCarlo.
        let mut bad = good.clone();
        bad.method = InferenceMethod::MonteCarlo;
        bad.n_permutations = 0;
        assert!(matches!(
            robinson_turner_two_sample_test(&[c(), c()], &[c(), c()], &bad),
            Err(RobinsonTurnerError::InvalidConfig(_))
        ));
        // try_from_rows validation.
        assert!(PointCloud::try_from_rows(vec![]).is_err());
        assert!(PointCloud::try_from_rows(vec![vec![]]).is_err());
        assert!(PointCloud::try_from_rows(vec![vec![0.0], vec![0.0, 1.0]]).is_err());
        assert!(PointCloud::try_from_rows(vec![vec![f64::NAN]]).is_err());
    }

    // 8. Combination overflow: checked label-space calculation errors rather
    //    than wrapping.
    #[test]
    fn combination_overflow() {
        assert_eq!(checked_binomial(4, 2), Some(6));
        assert_eq!(checked_binomial(6, 3), Some(20));
        // C(1_000_000, 500_000) vastly exceeds u128::MAX.
        assert_eq!(checked_binomial(1_000_000, 500_000), None);
    }

    // 9. Essential-class policy: H0 always has one essential component.
    //    Reject errors; Drop records the count per cloud.
    #[test]
    fn essential_policy() {
        let a = [c2(0.0), c2(0.1)];
        let b = [c2(1.0), c2(1.1)];
        let mut cfg = RobinsonTurnerConfig::new(0, 2.0);
        let err = robinson_turner_two_sample_test(&a, &b, &cfg);
        assert!(matches!(
            err,
            Err(RobinsonTurnerError::EssentialClasses {
                dim: 0,
                count: 1,
                ..
            })
        ));
        cfg.essential_class_policy = EssentialClassPolicy::Drop;
        let res = robinson_turner_two_sample_test(&a, &b, &cfg).unwrap();
        assert_eq!(res.dropped_essential_classes, vec![1, 1, 1, 1]);
        assert!(res.warnings.iter().any(|w| w.contains("essential")));
    }

    fn c2(x: f64) -> PointCloud {
        tiny_cloud(&[&[x, 0.0], &[x + 0.05, 0.0]])
    }

    // 10. Online summary vs a retained reference sequence.
    #[test]
    fn online_summary_matches_reference() {
        let data: Vec<f64> = (0..250).map(|i| ((i * 37) % 101) as f64 * 0.37).collect();
        let mut s = OnlineSummary::new();
        for &x in &data {
            s.push(x);
        }
        let got = s.finish();
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let sd = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(got.count, data.len() as u64);
        assert!((got.mean - mean).abs() < 1e-9);
        assert!((got.sample_std_dev - sd).abs() < 1e-9);
        assert_eq!(got.min, min);
        assert_eq!(got.max, max);
        // P² approximate quartiles within a documented tolerance of exact
        // type-7 quantiles (P² is approximate; tolerance is loose on purpose).
        let mut v = data.clone();
        v.sort_by(f64::total_cmp);
        let exact_q = |p: f64| {
            let h = p * (v.len() - 1) as f64;
            let lo = h.floor() as usize;
            let hi = (lo + 1).min(v.len() - 1);
            v[lo] + (h - lo as f64) * (v[hi] - v[lo])
        };
        let tol = sd * 0.5; // documented approximation tolerance
        assert!((got.approximate_p25 - exact_q(0.25)).abs() < tol);
        assert!((got.approximate_p50 - exact_q(0.5)).abs() < tol);
        assert!((got.approximate_p75 - exact_q(0.75)).abs() < tol);
        // Deterministic: same stream → same quantiles.
        let mut s2 = OnlineSummary::new();
        for &x in &data {
            s2.push(x);
        }
        assert_eq!(got, s2.finish());
    }

    // Matrix symmetry / zero diagonal / pairwise call counting via the seam.
    #[test]
    fn distance_matrix_once_and_symmetric() {
        __test_seams::reset_pairwise_distance_calls();
        let a = [c2(0.0), c2(0.1), c2(0.2)];
        let b = [c2(1.0), c2(1.1), c2(1.2)];
        let mut cfg = RobinsonTurnerConfig::new(0, 2.0);
        cfg.essential_class_policy = EssentialClassPolicy::Drop;
        cfg.return_distance_matrix = true;
        let res = robinson_turner_two_sample_test(&a, &b, &cfg).unwrap();
        let m = res.distance_matrix.unwrap();
        let n = m.len();
        assert_eq!(n, 6);
        for (i, row) in m.iter().enumerate() {
            assert_eq!(row[i], 0.0);
            for (j, &v) in row.iter().enumerate() {
                assert_eq!(v, m[j][i]);
            }
        }
        // Each upper-triangle pair computed exactly once.
        assert_eq!(__test_seams::pairwise_distance_calls(), n * (n - 1) / 2);
    }
}
