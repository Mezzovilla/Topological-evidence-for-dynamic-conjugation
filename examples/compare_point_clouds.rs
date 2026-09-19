use stattda::{
    topological_signature_test, EssentialClassPolicy, InferenceMethod, PointCloud,
    TopologicalSignatureConfig,
};

fn main() {
    let x = PointCloud::try_from_rows(vec![
        vec![1.0, 0.0], vec![0.5, 0.87], vec![-0.5, 0.87],
        vec![-1.0, 0.0], vec![-0.5, -0.87], vec![0.5, -0.87],
    ])
    .unwrap();
    let y = PointCloud::try_from_rows(vec![
        vec![0.0, 0.0], vec![0.4, 0.0], vec![0.8, 0.0], vec![1.2, 0.0],
    ])
    .unwrap();

    let mut config = TopologicalSignatureConfig::new(vec![0, 1], 8, 2.5);
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
}
