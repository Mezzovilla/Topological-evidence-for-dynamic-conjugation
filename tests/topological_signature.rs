//! Integration tests for `topological_signature_test`.
//!
//! The test lifts the Robinson–Turner primitive to a "one cloud per group"
//! workflow: `n_samples` sub-clouds are drawn (uniformly, with replacement)
//! from each input cloud, one permutation test runs per requested homology
//! dimension, and p-values are multiplicity-adjusted. Data are kept tiny
//! (a handful of 2D points) and inference is exact so tests stay fast and
//! deterministic.

use stattda::{
    EssentialClassPolicy, InferenceMethod, MultipleTestingCorrection, PointCloud,
    RobinsonTurnerConfig, TopologicalSignatureConfig, robinson_turner_two_sample_test,
    topological_signature_test,
};

fn cloud(rows: Vec<Vec<f64>>) -> PointCloud {
    PointCloud::try_from_rows(rows).expect("generated cloud must be valid")
}

/// Six points forming a rough ring (one prominent H1 loop).
fn ring_cloud() -> PointCloud {
    cloud(vec![
        vec![1.0, 0.0],
        vec![0.5, 0.87],
        vec![-0.5, 0.87],
        vec![-1.0, 0.0],
        vec![-0.5, -0.87],
        vec![0.5, -0.87],
    ])
}

/// Six points along a noisy segment (no H1 loop).
fn segment_cloud() -> PointCloud {
    cloud(vec![
        vec![0.0, 0.0],
        vec![0.2, 0.02],
        vec![0.4, -0.02],
        vec![0.6, 0.01],
        vec![0.8, -0.01],
        vec![1.0, 0.0],
    ])
}

/// Exact-inference config over `dims`, sampling `n` clouds per group.
/// `Drop` is set globally: it is required for H0 (one component always
/// survives a finite filtration) and is inert for higher dimensions.
fn config(dims: Vec<usize>, n: usize, seed: Option<u64>) -> TopologicalSignatureConfig {
    let mut cfg = TopologicalSignatureConfig::new(dims, n);
    cfg.max_edge_length = Some(2.5);
    cfg.method = InferenceMethod::Exact;
    cfg.max_exact_labelings = 100_000;
    cfg.random_seed = seed;
    cfg.essential_class_policy = EssentialClassPolicy::Drop;
    cfg
}

#[test]
fn basic_two_cloud_input_returns_aligned_result_vectors() {
    let (x, y) = (ring_cloud(), segment_cloud());
    let result = topological_signature_test(&x, &y, &config(vec![1], 4, Some(7)))
        .expect("signature test must succeed");

    let n = result.homology_dimensions.len();
    assert_eq!(result.homology_dimensions, vec![1]);
    assert_eq!(result.p_values.len(), n);
    assert_eq!(result.adjusted_p_values.len(), n);
    assert_eq!(result.reject_null.len(), n);
    assert_eq!(result.dimension_results.len(), n);
    assert_eq!(result.sampled_group_sizes, (4, 4));
    // sampled_cloud_size = None keeps each source cloud's point count.
    assert_eq!(result.sampled_point_counts, (6, 6));
    for &p in &result.p_values {
        assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
    }
    assert!(!result.interpretation.is_empty());
}

#[test]
fn multiple_dimensions_produce_one_result_each_in_caller_order() {
    let (x, y) = (ring_cloud(), segment_cloud());
    let result = topological_signature_test(&x, &y, &config(vec![0, 1], 3, Some(11)))
        .expect("signature test must succeed");

    assert_eq!(result.homology_dimensions, vec![0, 1]);
    assert_eq!(result.dimension_results.len(), 2);
    for (dim_result, &dim) in result.dimension_results.iter().zip(&[0usize, 1usize]) {
        assert_eq!(dim_result.homology_dim, dim);
    }
    assert_eq!(result.p_values.len(), 2);
    assert_eq!(result.adjusted_p_values.len(), 2);
}

#[test]
fn identical_config_with_fixed_seed_is_fully_reproducible() {
    let (x, y) = (ring_cloud(), segment_cloud());
    let mut cfg = config(vec![0, 1], 4, Some(0x5EED));
    cfg.method = InferenceMethod::MonteCarlo;
    cfg.n_permutations = 99;

    let r1 = topological_signature_test(&x, &y, &cfg).unwrap();
    let r2 = topological_signature_test(&x, &y, &cfg).unwrap();
    assert_eq!(r1, r2, "same config + fixed seed must reproduce the result");
    assert_eq!(r1.random_seed, 0x5EED);
}

#[test]
fn recorded_generated_seed_replays_full_result() {
    let (x, y) = (ring_cloud(), segment_cloud());
    let mut cfg = config(vec![1], 4, None);
    cfg.method = InferenceMethod::MonteCarlo;
    cfg.n_permutations = 99;

    let r1 = topological_signature_test(&x, &y, &cfg).unwrap();
    cfg.random_seed = Some(r1.random_seed);
    let r2 = topological_signature_test(&x, &y, &cfg).unwrap();
    assert_eq!(
        r1, r2,
        "replaying the recorded generated seed must reproduce the result"
    );
}

#[test]
fn holm_is_default_and_adjusted_p_values_dominate_raw() {
    let mut cfg = TopologicalSignatureConfig::new(vec![0, 1], 4);
    cfg.max_edge_length = Some(2.5);
    assert_eq!(
        cfg.multiple_testing_correction,
        MultipleTestingCorrection::Holm,
        "Holm must be the default correction"
    );
    assert!(cfg.sampled_cloud_size.is_none());

    let (x, y) = (ring_cloud(), segment_cloud());
    let result = topological_signature_test(&x, &y, &config(vec![0, 1], 4, Some(3))).unwrap();
    assert_eq!(
        result.multiple_testing_correction,
        MultipleTestingCorrection::Holm
    );
    for (adj, raw) in result.adjusted_p_values.iter().zip(&result.p_values) {
        assert!(
            adj >= raw,
            "Holm-adjusted p {adj} must be >= raw p {raw} componentwise"
        );
        assert!(*adj <= 1.0);
    }
}

#[test]
fn none_correction_leaves_adjusted_equal_to_raw() {
    let (x, y) = (ring_cloud(), segment_cloud());
    let mut cfg = config(vec![0, 1], 4, Some(5));
    cfg.multiple_testing_correction = MultipleTestingCorrection::None;

    let result = topological_signature_test(&x, &y, &cfg).unwrap();
    assert_eq!(
        result.multiple_testing_correction,
        MultipleTestingCorrection::None
    );
    assert_eq!(result.adjusted_p_values, result.p_values);
}

#[test]
fn readme_example_shape_compiles_and_succeeds() {
    // Mirrors the README "Topological signature test" example verbatim:
    // same clouds, same config values, same Monte-Carlo settings.
    let x = ring_cloud();
    let y = cloud(vec![
        vec![0.0, 0.0],
        vec![0.4, 0.0],
        vec![0.8, 0.0],
        vec![1.2, 0.0],
    ]);

    let mut config = TopologicalSignatureConfig::new(vec![0, 1], 8);
    config.max_edge_length = Some(2.5);
    config.method = InferenceMethod::MonteCarlo;
    config.n_permutations = 999;
    config.random_seed = Some(42);
    config.essential_class_policy = EssentialClassPolicy::Drop;

    let result = topological_signature_test(&x, &y, &config).expect("test failed");

    let n = result.homology_dimensions.len();
    assert_eq!(result.homology_dimensions, vec![0, 1]);
    assert_eq!(result.p_values.len(), n);
    assert_eq!(result.adjusted_p_values.len(), n);
    assert_eq!(result.reject_null.len(), n);
    assert_eq!(result.dimension_results.len(), n);
}

#[test]
fn dimension_order_does_not_change_global_or_per_dimension_results() {
    // Monte-Carlo runs with [0, 1] and [1, 0], same root seed and otherwise
    // identical config: per-dimension permutation streams are keyed by the
    // homology degree, so reordering the request changes only output order.
    let (x, y) = (ring_cloud(), segment_cloud());
    let mut cfg_ab = config(vec![0, 1], 4, Some(0x5EED));
    cfg_ab.method = InferenceMethod::MonteCarlo;
    cfg_ab.n_permutations = 99;
    let mut cfg_ba = cfg_ab.clone();
    cfg_ba.homology_dimensions = vec![1, 0];

    let ab = topological_signature_test(&x, &y, &cfg_ab).unwrap();
    let ba = topological_signature_test(&x, &y, &cfg_ba).unwrap();

    // Global combination is order-invariant.
    assert_eq!(ab.global_p_value, ba.global_p_value);
    assert_eq!(ab.global_test_statistic, ba.global_test_statistic);
    assert_eq!(ab.global_reject_null, ba.global_reject_null);
    assert_eq!(ab.global_combination_method, ba.global_combination_method);

    // Per-dimension results match after mapping by homology dimension.
    for dim in [0usize, 1usize] {
        let i = ab
            .homology_dimensions
            .iter()
            .position(|&d| d == dim)
            .unwrap();
        let j = ba
            .homology_dimensions
            .iter()
            .position(|&d| d == dim)
            .unwrap();
        assert_eq!(ab.p_values[i], ba.p_values[j], "dim {dim} raw p differs");
        assert_eq!(
            ab.adjusted_p_values[i], ba.adjusted_p_values[j],
            "dim {dim} adjusted p differs"
        );
        assert_eq!(
            ab.dimension_results[i].statistic, ba.dimension_results[j].statistic,
            "dim {dim} statistic differs"
        );
        assert_eq!(
            ab.dimension_results[i].p_value, ba.dimension_results[j].p_value,
            "dim {dim} result p-value differs"
        );
    }
}

#[test]
fn robinson_turner_two_sample_test_still_works_standalone() {
    // The lower-level primitive takes groups OF clouds directly.
    let group_a = vec![
        cloud(vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ]),
        cloud(vec![
            vec![0.0, 0.0],
            vec![1.2, 0.0],
            vec![1.2, 1.2],
            vec![0.0, 1.2],
        ]),
    ];
    let group_b = vec![
        cloud(vec![vec![0.0, 0.0], vec![0.3, 0.0], vec![0.6, 0.0]]),
        cloud(vec![vec![0.0, 0.0], vec![0.4, 0.0], vec![0.8, 0.0]]),
    ];
    let mut cfg = RobinsonTurnerConfig::new(1);
    cfg.max_edge_length = Some(2.5);
    cfg.method = InferenceMethod::Exact;

    let result = robinson_turner_two_sample_test(&group_a, &group_b, &cfg)
        .expect("primitive test must succeed");
    assert_eq!(result.group_sizes, (2, 2));
    assert!((0.0..=1.0).contains(&result.p_value));
}
