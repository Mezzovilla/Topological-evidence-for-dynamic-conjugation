# stattda

Statistical tests for topological data analysis. Currently this crate provides
the Robinson–Turner two-sample test: a permutation test that compares two
independent groups of point clouds through their `k`-dimensional persistence
diagrams (Robinson & Turner, *Hypothesis Testing for Topological Data
Analysis*, arXiv:1310.7467v2, 2016, Sections 3–4).

## Example

```rust
use stattda::{
    robinson_turner_two_sample_test, DiagramDistance, InferenceMethod, PointCloud,
    RobinsonTurnerConfig,
};

// Group A: three clouds, each a noisy-ish square (one prominent loop).
let group_a: Vec<PointCloud> = [1.0_f64, 1.1, 0.9]
    .iter()
    .map(|&s| {
        PointCloud::try_from_rows(vec![
            vec![0.0, 0.0],
            vec![s, 0.0],
            vec![s, s],
            vec![0.0, s],
        ])
        .unwrap()
    })
    .collect();

// Group B: three clouds sampled near a line segment (no loop).
let group_b: Vec<PointCloud> = [0.0_f64, 0.05, -0.05]
    .iter()
    .map(|&j| {
        PointCloud::try_from_rows(vec![
            vec![0.0, j],
            vec![0.4, j],
            vec![0.8, j],
            vec![1.2, j],
        ])
        .unwrap()
    })
    .collect();

// homology_dim = 1 (loops); explicit filtration cut-off of 2.0
// (None — the default — uses the maximum within-cloud diameter instead).
let mut config = RobinsonTurnerConfig::new(1);
config.max_edge_length = Some(2.0);
config.diagram_distance = DiagramDistance::Wasserstein2;
config.method = InferenceMethod::MonteCarlo;
config.n_permutations = 999;
config.random_seed = Some(42); // deterministic run

let result = robinson_turner_two_sample_test(&group_a, &group_b, &config)
    .expect("two-sample test failed");

println!("F statistic = {}", result.statistic);
println!("p-value     = {}", result.p_value);
println!("reject H0   = {}", result.reject_null);
```

Expected output:

```text
F statistic = 0.020000000000000014
p-value     = 0.089
reject H0   = false
```

This example is mirrored and executed as an integration test in
`tests/public_example.rs`. It is also runnable as a cargo example:

```text
cargo run --example public_example
```

## Topological signature test

`topological_signature_test` is a higher-level entry point for the common
"one cloud per group" design: instead of supplying groups *of* clouds, you
supply two input point clouds `X` and `Y`. The function draws `n_samples`
sub-clouds from each input (uniformly, with replacement; `sampled_cloud_size`
defaults to `None`, which keeps each source cloud's point count), then runs
one Robinson–Turner test per requested homology dimension and reports raw and
multiplicity-adjusted p-values.

```rust
use stattda::{
    topological_signature_test, EssentialClassPolicy, InferenceMethod, PointCloud,
    TopologicalSignatureConfig,
};

let x = PointCloud::try_from_rows(vec![
    vec![1.0, 0.0], vec![0.5, 0.87], vec![-0.5, 0.87],
    vec![-1.0, 0.0], vec![-0.5, -0.87], vec![0.5, -0.87],
])
.unwrap();
let y = PointCloud::try_from_rows(vec![
    vec![0.0, 0.0], vec![0.4, 0.0], vec![0.8, 0.0], vec![1.2, 0.0],
])
.unwrap();

let mut config = TopologicalSignatureConfig::new(vec![0, 1], 8);
config.max_edge_length = Some(2.5);
config.method = InferenceMethod::MonteCarlo;
config.n_permutations = 999;
config.random_seed = Some(42); // deterministic run; None uses OsRng
// H0 keeps one essential component under a finite filtration: censor it.
config.essential_class_policy = EssentialClassPolicy::Drop;

let result = topological_signature_test(&x, &y, &config).expect("test failed");
for ((&dim, &p), &adj) in result
    .homology_dimensions
    .iter()
    .zip(&result.p_values)
    .zip(&result.adjusted_p_values)
{
    println!("H{dim}: raw p = {p}, adjusted p = {adj}");
}
```

Expected output:

```text
H0: raw p = 0.001, adjusted p = 0.002
H1: raw p = 0.455, adjusted p = 0.455
```

This example is also runnable as a cargo example:

```text
cargo run --example compare_point_clouds
```

*Interpretation.* There is one test per homology dimension in
`homology_dimensions`; `p_values` holds the raw per-dimension p-values and
`adjusted_p_values` holds them after `multiple_testing_correction`
(`Holm` by default, `None` to opt out). A rejection at `alpha` is evidence of
differences between the topological distributions induced by `X` and `Y`
under the configured sampling / filtration / metric / test pipeline. It is
*not* a claim about any individual feature, and **non-rejection is not proof
that the two distributions are equal**. The `random_seed` used (given or
generated) is recorded in the result; replaying it reproduces the full run.

## Synthetic manifold examples

The synthetic dataset generators can be used directly with
`topological_signature_test`. Two runnable examples compare independently
sampled point clouds from spheres, tori, and Klein bottles:

```text
cargo run --example compare_same_manifolds
cargo run --example compare_distinct_manifolds
```

Both examples use fixed seeds, `H1`, exact permutation inference, and
`EssentialClassPolicy::Drop` to censor classes still alive at the cut-off.
`compare_same_manifolds` samples 1,000 source points per cloud and draws 10
resampled clouds of 40 points per group; `compare_distinct_manifolds` samples
100 source points per cloud and draws 6 resampled clouds of 40 points per
group. Both omit `max_edge_length` (constructor default `None`), so for each
comparison the cut-off is resolved automatically as the maximum internal
Euclidean diameter of the two source clouds and recorded in
`ph_configuration.max_edge_length`.

The deterministic runs produce:

| Comparison | Adjusted H1 p-value | Automatic cutoff | Reject at 0.05? |
| --- | ---: | ---: | :---: |
| Sphere vs. sphere | 0.673310 | 1.999999 | No |
| Torus vs. torus | 0.784408 | 1.999756 | No |
| Klein bottle vs. Klein bottle | 0.812163 | 2.118385 | No |
| Sphere vs. torus | 0.002165 | 1.999911 | Yes |
| Sphere vs. Klein bottle | 0.006494 | 2.096112 | Yes |
| Torus vs. Klein bottle | 0.857143 | 2.096112 | No |

These runs found no evidence against exchangeability of the configured `H1`
signatures for four of the six pairs, and rejections for sphere vs. torus and
sphere vs. Klein bottle at `alpha = 0.05`. Neither direction should be
over-read: rejection is evidence about the configured sampling / filtration /
metric / test pipeline, not a certificate that the manifolds differ
topologically, and non-rejection is not proof of equality. Power and outcomes
depend on the finite samples, resampling size, the (here automatically
resolved) filtration cut-off, censoring policy, and homology dimensions;
automatic cutoff selection does not resolve the pseudoreplication limitation
of this resampling design. The values above are reproducible usage examples, not
benchmark results.

## What the test claims (and what it does not)

*Null hypothesis.* Conditional on the configured point-cloud-to-diagram
pipeline (homology degree, filtration cut-off, metric, censoring policy), the
pooled persistence diagrams are *exchangeable* across the two group labels —
i.e. the two groups induce the same distribution of `k`-dimensional
persistence diagrams.

*Lower-tail interpretation.* The statistic `F_{p,q}` measures within-group
dispersion. A grouping that "fits" has unusually *small* within-group
dispersion, so the p-value is the lower-tail proportion of permutation losses
at or below the observed loss. A small p-value is evidence against label
exchangeability. A large p-value is *not* evidence that the distributions are
equal.

*Caveats.* The result says nothing about the significance of any individual
topological feature in any individual cloud, and it does not establish a
causal group effect. Diagrams depend on the filtration construction and on the
scale of the data — this test does not provide filtration-independent topology
claims.

## The statistic

For labeling `L = (G_1, G_2)` with group sizes `n_m`, diagram-distance order
`p`, and loss exponent `q`:

```text
F_{p,q}(L) = sum_{m=1}^{2}  1 / (2 n_m (n_m - 1))  sum_i sum_j  W_p(D_{m,i}, D_{m,j})^q
```

Two distinct exponents appear and must not be conflated:

* `p` — the **Wasserstein order** of the diagram distance (`DiagramDistance`:
  1, 2, or ∞ for bottleneck);
* `q` — the **loss exponent** applied to each pairwise distance (1 or 2).

The default is `F_{2,2}` (Robinson & Turner's recommended choice).

*Ground metric.* All diagram distances use the `L-infinity` norm in the
birth–death plane, matching the pinned backend's convention:

```text
W_p(X, Y) = ( min_phi  sum_x ||x - phi(x)||_inf^p )^(1/p)
```

with matching to the diagonal allowed. Finite `p` changes how matched costs
accumulate — it does *not* change the ground norm.

## Why permute clouds, not points

Each point cloud is one independent observation producing exactly one
persistence diagram. The null hypothesis concerns the *distribution of
diagrams per group*, so permutations act at the cloud level: a labeling
reassigns whole clouds (whole diagrams) to groups. Raw points are never pooled
or resampled across clouds, and permuting points inside a cloud would destroy
the very geometry the diagrams summarize.

## Exact versus Monte-Carlo inference

There are `C(N, n1)` label-preserving allocations of `N = n1 + n2` clouds.

* *Exact*: when `C(N, n1) <= max_exact_labelings` (and `InferenceMethod::Auto`
  or `Exact`), every allocation is enumerated and
  `p_value = extreme_count / C(N, n1)`.
* *Monte Carlo*: otherwise `B` independent uniform allocations are drawn and
  the corrected p-value `(b_random + 1) / (B + 1)` is reported. The `+1`
  correction counts the observed allocation itself, so the p-value is never
  zero; with the default `B = 9_999` the resolution `1 / (B + 1)` is `1e-4`
  and is reported as `p_value_resolution` (a warning is emitted when it
  exceeds `alpha`).

Monte-Carlo sampling uses `ChaCha8Rng` with a recorded `u64` seed, so runs are
reproducible on the same supported crate versions and target.

## Finite filtrations and essential classes

The Rips filtration is truncated at `max_edge_length` and built through
simplex dimension `k + 1`. `max_edge_length` is an `Option<f64>`: `None` (the
default) resolves it automatically as the maximum within-cloud Euclidean
diameter of the supplied clouds — scale-dependent, `O(sum n_i^2 d)` over
clouds, and potentially a dense complex — while `Some(value)` (finite,
strictly positive) is the cost-bounding override. A `k`-dimensional class
still alive at the cut-off is *essential* for this analysis — its death time
is unknown. `EssentialClassPolicy::Reject` (default) returns a structured
error; `EssentialClassPolicy::Drop` explicitly censors those classes and
records how many were removed per cloud. `H_0` analyses typically need `Drop`
because one connected component always survives the filtration.

## Computational scaling

* One Rips persistent-homology computation **per cloud** (the backend
  enumerates simplices, so keep clouds modest).
* `O(N^2)` diagram-distance evaluations — one symmetric `N x N` matrix,
  computed once, with an exact zero diagonal.
* All permutation work (exact enumeration or Monte-Carlo draws) operates on
  the precomputed matrix only; persistent homology and diagram distances are
  never recomputed inside a permutation.

## Backend

Persistent homology and optimal diagram distances (Wasserstein-1,
Wasserstein-2, bottleneck — all with diagonal matching and optimal, not
greedy, assignment) come from `oxicuda-tda`, pinned exactly to `=0.5.5`. That
crate is pre-1.0 and describes itself as alpha; it is isolated behind an
internal adapter, its types never appear in this public API, and its version
is recorded in every result (`backend`, `backend_version`). Integration tests
in `tests/backend_integration.rs` pin down the behaviors relied upon —
including an adversarial diagram pair where greedy matching would differ from
the optimal value.

*Serialization note.* `RobinsonTurnerTestResult` derives
`serde::Serialize`. The optional `distance_matrix` field serializes as `null`
(or is omitted entirely if the `skip_serializing_if` annotation is used — see
`tests/serialization.rs`, which pins whichever convention is implemented).

## Verification

```text
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo doc --no-deps
```
