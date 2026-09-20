//! Compile-and-run mirror of the README usage example. Keep this in sync with
//! the `README.md` example block: it proves the documented public example
//! actually compiles and executes against the real API.

use stattda::{
    DiagramDistance, InferenceMethod, PointCloud, RobinsonTurnerConfig,
    robinson_turner_two_sample_test,
};

#[test]
fn readme_example_runs() {
    // Group A: three clouds, each a noisy-ish square (one prominent loop).
    let group_a: Vec<PointCloud> = [1.0_f64, 1.1, 0.9]
        .iter()
        .map(|&s| {
            PointCloud::try_from_rows(vec![vec![0.0, 0.0], vec![s, 0.0], vec![s, s], vec![0.0, s]])
                .unwrap()
        })
        .collect();

    // Group B: three clouds sampled near a line segment (no loop).
    let group_b: Vec<PointCloud> = [0.0_f64, 0.05, -0.05]
        .iter()
        .map(|&j| {
            PointCloud::try_from_rows(vec![vec![0.0, j], vec![0.4, j], vec![0.8, j], vec![1.2, j]])
                .unwrap()
        })
        .collect();

    // homology_dim = 1 (loops), max_edge_length = 2.0 (filtration cut-off).
    let mut config = RobinsonTurnerConfig::new(1);
    config.max_edge_length = Some(2.0);
    config.diagram_distance = DiagramDistance::Wasserstein2;
    config.method = InferenceMethod::MonteCarlo;
    config.n_permutations = 999;
    config.random_seed = Some(42); // deterministic run

    let result = robinson_turner_two_sample_test(&group_a, &group_b, &config)
        .expect("two-sample test failed");

    assert!(result.statistic.is_finite());
    assert!(result.p_value > 0.0 && result.p_value <= 1.0);
    assert_eq!(result.group_sizes, (3, 3));
    assert_eq!(result.random_seed, Some(42));
}
