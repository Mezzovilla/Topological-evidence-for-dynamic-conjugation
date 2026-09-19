# Implement the Robinson–Turner two-sample test as a Rust library

## Summary

Implement a pure Rust library function named `robinson_turner_two_sample_test` that compares two independent collections of point clouds through their `k`-dimensional persistence diagrams. It must return the observed within-group dispersion statistic, a valid exact or Monte-Carlo permutation p-value, and enough provenance to reproduce the analysis.

The statistical method is the two-sample randomization test in Robinson & Turner, *Hypothesis Testing for Topological Data Analysis*, arXiv:1310.7467v2 (2016), Sections 3–4, especially equations (1) and (4) and Algorithm 2.

The scientific target is:

> Test whether the two groups induce the same distribution of `k`-dimensional persistence diagrams under one fixed point-cloud-to-diagram pipeline.

A small p-value is evidence against exchangeability of group labels for the selected topological summary and preprocessing choices. It does not prove that an individual point cloud has a significant topological feature, and it does not establish a causal group effect.

## Scope

### In scope

- A Rust library crate; no Python runtime, bindings, or extension module.
- Two independent groups of point clouds in `R^d`, with any positive ambient dimension supported by the input representation.
- One fixed homology degree per test, subject to the selected backend's documented filtration limit.
- Euclidean Vietoris–Rips persistent homology over `Z/2`.
- Finite persistence diagrams and optimal diagram distances with diagonal matching.
- Robinson and Turner's within-group joint-loss statistic `F_{p,q}`.
- Exact enumeration for small label spaces and Monte-Carlo randomization otherwise.
- The corrected Monte-Carlo p-value `(b_random + 1) / (B + 1)`.
- Deterministic seeded results, Rust tests, examples, documentation, structured errors, and a Serde-serializable result.

### Explicit non-goals

- No one-sample significance test for a feature in one diagram.
- No parametric t-test, normal approximation, or test on persistence counts.
- No Fréchet means inside permutations.
- No pooling raw points across clouds: every cloud is one independent observation and yields exactly one diagram.
- No custom persistent-homology reduction or custom optimal-assignment solver.
- No covariate adjustment, repeated-measures design, batch correction, or restricted permutation scheme.
- No CLI, argument parser, file loader, Python API, WebAssembly API, or automatic file output.

## Rust backend decision

### Investigated crate: `tda`

The `tda` crate version `0.1.0` provides Vietoris–Rips persistent homology, persistence diagrams, and public `wasserstein_distance` and `bottleneck_distance` functions. It is not suitable as the inferential distance backend for this implementation: inspection of its published source shows that both distances use greedy matching, despite their documentation describing an infimum over matchings. A greedy result is not guaranteed to be the optimal persistence-diagram metric required by the Robinson–Turner statistic.

Do not use `tda::persistence_diagram::{wasserstein_distance, bottleneck_distance}` for this test unless a later pinned release is independently verified to perform optimal matching and regression-tested against trusted values.

References inspected:

- <https://docs.rs/tda/0.1.0/tda/>
- <https://docs.rs/tda/0.1.0/src/tda/persistence_diagram.rs.html>

### Selected crate: `oxicuda-tda`

Pin `oxicuda-tda = "=0.5.5"`. This release provides:

- `Filtration::vietoris_rips_from_points` for Euclidean Rips filtrations;
- boundary-matrix reduction and persistence-pair extraction over `Z/2`;
- exact bottleneck distance through threshold search and bipartite matching;
- general finite `p`-Wasserstein distance through an augmented cost matrix and Hungarian/Jonker–Volgenant-style assignment;
- diagonal matching and valid empty-diagram behavior.

The crate is pre-1.0 and describes itself as alpha. Isolate it behind a small internal adapter, pin it exactly, record its version in every result, and add integration tests around every relied-upon behavior. Do not expose its types in the public API.

References inspected:

- <https://docs.rs/oxicuda-tda/0.5.5/oxicuda_tda/>
- <https://docs.rs/oxicuda-tda/0.5.5/oxicuda_tda/complex/filtration/struct.Filtration.html>
- <https://docs.rs/oxicuda-tda/0.5.5/oxicuda_tda/persistence/wasserstein_p/fn.wasserstein_p.html>
- <https://docs.rs/oxicuda-tda/0.5.5/oxicuda_tda/persistence/distance/fn.bottleneck_distance.html>

### Metric convention

Use the convention implemented and documented by `oxicuda-tda`:

```text
W_p(X,Y) = (min_phi sum_x ||x - phi(x)||_inf^p)^(1/p)
```

for finite `p`, with diagonal matching, and `W_inf` for bottleneck distance. Here `p` is the Wasserstein order and the ground metric is `L-infinity` in the birth–death plane. Support `p = 1`, `p = 2` (default), and `p = infinity`.

This convention must be explicit in API documentation and result provenance. Do not describe finite `p` as changing the ground norm.

## Statistical design

Let `A = (A_1, ..., A_n1)` and `B = (B_1, ..., B_n2)` be groups of point clouds. Clouds may have different point counts, but all must have the same positive ambient dimension. For fixed `k`, map each cloud independently to one requested-degree persistence diagram:

```text
D_i = PH_k(cloud_i)
```

The test assumes independent cloud-level observations whose labels are exchangeable under the null:

- `H0`: the pooled diagrams are identically distributed, conditional on the configured pipeline;
- `H1`: their distributions differ.

The test is two-sample/two-sided in its scientific target but lower-tailed in the loss: a meaningful supplied grouping is expected to have unusually small within-group dispersion.

### Truncation and essential classes

Require a finite, strictly positive `max_edge_length`. Build each Rips filtration through simplex dimension `homology_dim + 1`, because `(k+1)`-simplices are needed to kill `k`-classes. `oxicuda-tda 0.5.5` supports filtration dimensions through `6`, so this adapter supports `homology_dim <= 5`; reject larger values before backend work.

Expose:

```rust
pub enum EssentialClassPolicy {
    Reject,
    Drop,
}
```

Default to `Reject`. If the requested-degree diagram contains an essential class, return a structured error explaining that the filtration ended before the class died. `Drop` is an explicit opt-in censoring policy that removes requested-degree essential classes and records the count removed for every cloud. Never silently remove them. Document that `H_0` commonly requires `Drop` because one connected component remains essential.

### Joint-loss statistic

For labeling `L = (G_1, G_2)`, group sizes `n_m`, diagram-distance order `p`, and loss exponent `q`, calculate:

```math
F_{p,q}(L) = \sum_{m=1}^{2}
  \frac{1}{2 n_m (n_m - 1)}
  \sum_{i=1}^{n_m}\sum_{j=1}^{n_m}
  W_p(D_{m,i}, D_{m,j})^q.
```

Requirements:

- Require `n1 >= 2` and `n2 >= 2`.
- Default to `F_{2,2}`.
- Support `q in {1, 2}` and reject all other values.
- Sum `i < j` and divide by `n_m(n_m-1)`, which is algebraically equivalent to the displayed formula.
- Precompute one symmetric `N x N` diagram-distance matrix, force an exact zero diagonal and mirrored entries, and reuse it for every labeling.
- Never recompute persistent homology or diagram distances inside a permutation.
- Treat values within `tie_tolerance` as ties and count ties conservatively: `F_perm <= F_observed + tie_tolerance`.

### Exact and Monte-Carlo modes

There are `C(N, n1)` allocations preserving the supplied group sizes.

#### Exact

When `C(N, n1) <= max_exact_labelings` and the method is `Auto`, enumerate all combinations, including the observed allocation. For `Exact`, return an error rather than exceed `max_exact_labelings`; callers may deliberately raise that guard.

```text
p_value = extreme_count / C(N, n1)
```

Set `n_random_permutations = 0` in exact mode.

#### Monte Carlo

Otherwise, draw `B` independent uniformly random allocations. Do not exclude or specially handle the observed allocation if it appears among random draws. For every draw, shuffle a fresh `0..N` index vector and take its first `n1` indices.

```text
b_random = number of random losses <= observed + tolerance
extreme_count = b_random + 1
p_value = (b_random + 1) / (B + 1)
```

Default to `B = 9_999`; the minimum p-value is then `0.0001`. Reject `B = 0` when Monte Carlo is selected. Report resolution `1/(B+1)` and warn through the result when that resolution exceeds `alpha`.

Use `rand_chacha::ChaCha8Rng` and an explicit `u64` seed. If the caller supplies no seed, generate one from `rand::rngs::OsRng`, record the generated seed, and initialize `ChaCha8Rng` from it. Re-running with the recorded seed must reproduce the sampled allocations and every scalar/summary field on the same supported crate versions and target.

Maintain permutation summaries online rather than retaining every loss. Use Welford updates for count, mean, and sample standard deviation; track minimum and maximum; use deterministic P² estimators for approximate 25th, 50th, and 75th percentiles. Mark those quantiles as approximate in field names and documentation. The exact p-value itself must never depend on a quantile estimator.

## Public Rust API

The crate name remains `stattda`. Expose this API from `src/lib.rs`:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct PointCloud {
    pub coordinates: Vec<f64>, // row-major
    pub n_points: usize,
    pub ambient_dim: usize,
}

impl PointCloud {
    pub fn try_from_rows(rows: Vec<Vec<f64>>) -> Result<Self, RobinsonTurnerError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DiagramDistance {
    Wasserstein1,
    Wasserstein2,
    Bottleneck,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum InferenceMethod {
    Auto,
    Exact,
    MonteCarlo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum EssentialClassPolicy {
    Reject,
    Drop,
}

#[derive(Clone, Debug)]
pub struct RobinsonTurnerConfig {
    pub homology_dim: usize,
    pub max_edge_length: f64,
    pub diagram_distance: DiagramDistance,
    pub loss_q: u32,
    pub max_exact_labelings: u128,
    pub n_permutations: u64,
    pub method: InferenceMethod,
    pub random_seed: Option<u64>,
    pub alpha: f64,
    pub tie_tolerance: f64,
    pub essential_class_policy: EssentialClassPolicy,
    pub return_distance_matrix: bool,
}

impl Default for RobinsonTurnerConfig {
    fn default() -> Self; // defaults except homology_dim/max_edge_length must be documented
}

pub fn robinson_turner_two_sample_test(
    group_a: &[PointCloud],
    group_b: &[PointCloud],
    config: &RobinsonTurnerConfig,
) -> Result<RobinsonTurnerTestResult, RobinsonTurnerError>;
```

Because `homology_dim` and `max_edge_length` have no scientifically universal defaults, also provide a constructor that requires them:

```rust
impl RobinsonTurnerConfig {
    pub fn new(homology_dim: usize, max_edge_length: f64) -> Self;
}
```

`Default` may exist for Rust ergonomics, but its documentation must direct scientific callers to `new`; use `homology_dim = 1` and `max_edge_length = 1.0` only as mechanical defaults.

### Result and errors

Return an owned, immutable-by-default value deriving `Clone`, `Debug`, `PartialEq`, and `serde::Serialize`:

```rust
pub struct RobinsonTurnerTestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub reject_null: bool,
    pub alpha: f64,
    pub homology_dim: usize,
    pub diagram_distance: DiagramDistance,
    pub ground_metric: String,
    pub loss_q: u32,
    pub group_sizes: (usize, usize),
    pub inference_mode: InferenceMode,
    pub n_labelings_total: u128,
    pub n_random_permutations: u64,
    pub extreme_count: u64,
    pub p_value_resolution: f64,
    pub random_seed: Option<u64>,
    pub backend: String,
    pub backend_version: String,
    pub ph_configuration: PhConfiguration,
    pub diagram_sizes: Vec<usize>,
    pub dropped_essential_classes: Vec<usize>,
    pub permutation_summary: PermutationStatisticSummary,
    pub warnings: Vec<String>,
    pub interpretation: String,
    pub distance_matrix: Option<Vec<Vec<f64>>>,
}
```

Use typed serializable structs/enums for `InferenceMode`, `PhConfiguration`, and `PermutationStatisticSummary`; do not use unstructured maps. `extreme_count` means exact `b` in exact mode and `b_random + 1` in Monte-Carlo mode. In exact mode `random_seed` is `None`, because randomness is not used.

Define `RobinsonTurnerError` with `thiserror` and variants for invalid groups, malformed clouds, inconsistent dimensions, nonfinite coordinates, invalid configuration, label-space overflow, essential classes, and backend failure. Preserve the backend error as a source where practical; do not panic on user input or backend errors.

The interpretation is mechanically derived:

- `p_value < alpha`: “Reject label exchangeability at alpha; the groups have evidence of different k-dimensional persistence-diagram distributions under this analysis pipeline.”
- otherwise: “Do not reject label exchangeability at alpha; this is not evidence that the distributions are equal.”

Always append that the statistic concerns group differences in the selected persistent-homology summary, not feature significance for an individual cloud.

## Dependencies and project conversion

Convert the current Python scaffold into a standard Rust library crate:

- create `Cargo.toml` and `Cargo.lock`;
- use Rust edition 2024 if the installed stable toolchain supports it, otherwise edition 2021;
- remove Python packaging files and the Python placeholder module only as part of the approved implementation change;
- do not add a binary target;
- pin `oxicuda-tda = "=0.5.5"` exactly;
- use compatible, non-floating releases of `serde` with derive, `thiserror`, `rand`, `rand_chacha`, and `itertools` as needed;
- place library code under `src/` and integration tests under `tests/`.

The lockfile is required and must be committed with the implementation. Do not weaken Cargo security policy to resolve dependencies.

## Implementation plan

1. **Rust crate and public types**
   - Convert the scaffold to a library-only Cargo package.
   - Implement public input/config/result/error types and exports.
   - Keep all `oxicuda-tda` types internal.

2. **Validation**
   - Validate group sizes, rectangular nonempty clouds, finite coordinates, common positive ambient dimension, supported homology dimension, finite positive edge length, `q`, `alpha`, tolerance, and permutation settings.
   - Calculate `C(N,n1)` with checked `u128` arithmetic and return an overflow error.

3. **Persistent-homology adapter**
   - Build one Euclidean Rips filtration per cloud through dimension `k+1`.
   - Reduce through the documented `oxicuda-tda` boundary-matrix API before extracting pairs.
   - Select exactly degree `k` and enforce the essential-class policy.
   - Record filtration and backend provenance.

4. **Diagram-distance matrix**
   - Delegate finite distances to `oxicuda_tda::persistence::wasserstein_p::wasserstein_p` for orders 1 and 2 and to `oxicuda_tda::persistence::distance::bottleneck_distance` for infinity.
   - Compute only the upper triangle, mirror each value, and force zero diagonal.
   - Keep execution serial initially; parallelism is out of scope until deterministic equivalence is tested.

5. **Statistic and permutation engine**
   - Keep `within_group_loss` independent from PH extraction.
   - Enumerate exact combinations lazily with `itertools` or an internal iterator, never materializing all allocations.
   - Shuffle a fresh pooled-index vector per Monte-Carlo draw.
   - Update extreme counts and online summary state in one pass.

6. **Documentation and serialization**
   - Derive Serde serialization for result/provenance types.
   - Do not write files or force a JSON dependency in the runtime API; use `serde_json` only as a dev-dependency to test serializability.
   - Document assumptions, interpretation, backend alpha status, and metric convention.

## Required tests

### Unit tests

1. **Pairwise-loss identity**: for synthetic scalar distances `D[i,j] = |x_i-x_j|`, verify `F_{2,2}` equals the sum of the groups' unbiased sample variances for tested partitions.
2. **Hand-computed statistic**: verify a fixed `4 x 4` matrix and two groups of size two.
3. **Tie handling**: an all-zero matrix gives exact p-value `1.0` and counts every labeling.
4. **Exact enumeration**: `N=4, n1=2` visits six allocations and includes the observed allocation.
5. **Monte-Carlo correction**: when no random draw is extreme, p-value equals `1/(B+1)` and is nonzero.
6. **Seed reproducibility**: repeated Monte-Carlo calls with the same seed produce equal result fields and summaries.
7. **Input validation**: cover small groups, ragged/empty clouds, inconsistent dimensions, NaN/infinity, invalid homology dimension, edge length, `q`, alpha, tolerance, and permutation count.
8. **Combination overflow**: checked label-space calculation returns a structured error rather than wrapping.
9. **Essential policy**: `Reject` reports an error; `Drop` records exactly how many classes were removed.
10. **Online summary**: compare count/min/max/mean/sample-SD against a retained reference sequence and test deterministic approximate quantiles within documented tolerance.
11. **Serialization**: a result serializes through `serde_json`; omitted distance matrix remains `null` or absent according to the chosen Serde annotation and is documented.

### Backend integration tests

1. Empty versus empty and empty versus nonempty diagrams produce finite documented distances.
2. Orders 1 and 2 and bottleneck dispatch to the intended `oxicuda-tda` functions.
3. Hand-specified diagrams match direct pinned-backend calls.
4. Include an adversarial diagram pair where greedy matching differs from optimal matching and assert the backend result equals the optimal expected value.
5. A simple point cloud produces one selected-degree diagram without pooling across clouds.
6. Requested-degree essential classes are surfaced before policy handling.
7. The distance matrix is symmetric, has exact zero diagonal, and invokes each pairwise distance only once through a test seam.

### End-to-end deterministic tests

Keep datasets modest because the backend enumerates Rips simplices.

1. **Null smoke test**: repeatedly split clouds generated from one noisy-circle process and verify the rejection fraction is broadly compatible with the nominal level using a pre-specified, non-flaky tolerance.
2. **Alternative sensitivity smoke test**: noisy one-circle clouds versus noisy two-concentric-circle clouds; with pre-specified seeds and `homology_dim=1`, median alternative p-value is lower than median null p-value.
3. **Affine-scale behavior**: uniform scaling changes raw diagram distances/statistic; document that the default analysis is not scale invariant.
4. **Public example**: the README example compiles as an integration/doc test and calls `robinson_turner_two_sample_test`.

## Verification commands

The implementation is complete only when these pass on stable Rust:

```text
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo doc --no-deps
```

Run `cargo deny check` only if `cargo-deny` configuration is deliberately added and reviewed; do not invent a security-policy exception to make it pass.

## Documentation requirements

The crate-level docs and README must include:

- a runnable two-group in-memory point-cloud example using `robinson_turner_two_sample_test`;
- the null hypothesis and lower-tail interpretation;
- the formula for `F_{p,q}`, distinguishing Wasserstein order `p` from loss exponent `q`;
- the `L-infinity` ground-metric convention;
- why permutations occur at cloud level rather than point level;
- exact versus Monte-Carlo behavior and the nonzero correction;
- finite-filtration and essential-class policy;
- computational scaling: PH per cloud, `O(N^2)` diagram distances, and matrix-only permutation work;
- the selected crate's pre-1.0/alpha status and exact version pin;
- a statement that diagrams depend on filtration and scale, rather than declaring filtration-independent topology.

## Acceptance criteria

- [ ] The repository is a library-only Rust crate with no Python runtime or binary target.
- [ ] The public function is named `robinson_turner_two_sample_test`.
- [ ] It implements Robinson and Turner's `F_{p,q}` with default `F_{2,2}`.
- [ ] Every point cloud yields exactly one requested-degree diagram; raw points are never pooled across clouds.
- [ ] Persistent homology and optimal diagram distances come from pinned `oxicuda-tda 0.5.5` behind an internal adapter.
- [ ] The inspected `tda 0.1.0` greedy diagram-distance routines are not used for inference.
- [ ] The full diagram-distance matrix is built once and reused for all labelings.
- [ ] Exact mode returns the lower-tail proportion over all allocations.
- [ ] Monte-Carlo mode returns `(b_random+1)/(B+1)` and never zero.
- [ ] Recorded seeds reproduce Monte-Carlo results on the same supported environment.
- [ ] Empty diagrams, ties, invalid inputs, minimal groups, and essential classes have explicit tested behavior.
- [ ] The typed result records statistics, configuration, backend version, censoring, warnings, and provenance and derives Serde serialization.
- [ ] Unit, backend integration, deterministic end-to-end, formatting, Clippy, and documentation checks pass.
- [ ] Documentation states the statistical claim without presenting it as individual-feature significance or causality.

## References

- Andrew Robinson and Katharine Turner, “Hypothesis Testing for Topological Data Analysis,” arXiv:1310.7467v2, 2016: <https://arxiv.org/html/1310.7467v2>
- `tda 0.1.0` documentation and published source: <https://docs.rs/tda/0.1.0/tda/>
- `oxicuda-tda 0.5.5` documentation: <https://docs.rs/oxicuda-tda/0.5.5/oxicuda_tda/>
