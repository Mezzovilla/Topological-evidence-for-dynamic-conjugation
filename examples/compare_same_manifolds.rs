//! Compares two point clouds sampled independently from the same manifold
//! family. Because both clouds share the same underlying topology, the
//! topological signature test should generally not reject the null hypothesis.
//!
//! Run with: `cargo run --example compare_same_manifolds`

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use stattda::datasets::{KleinBottle, PointCloudGenerator, Sphere, Torus};
use stattda::{
    EssentialClassPolicy, InferenceMethod, PointCloud, TopologicalSignatureConfig,
    topological_signature_test,
};

const SOURCE_POINTS: usize = 1000;
const SAMPLED_POINTS: usize = 40;
const N_SAMPLES: usize = 10;

fn compare(label: &str, x: &PointCloud, y: &PointCloud, test_seed: u64) {
    // `max_edge_length` is left at its `None` default: the test resolves it
    // automatically as the maximum within-cloud Euclidean diameter.
    let mut config = TopologicalSignatureConfig::new(vec![1], N_SAMPLES);
    config.sampled_cloud_size = Some(SAMPLED_POINTS);
    config.method = InferenceMethod::Exact;
    config.essential_class_policy = EssentialClassPolicy::Drop;
    config.random_seed = Some(test_seed);
    let result =
        topological_signature_test(x, y, &config).expect("topological signature test failed");
    println!("{label}");
    println!("  H1 p-value: {:.6}", result.adjusted_p_values[0]);
    println!(
        "  automatic max edge length: {:.6}",
        result.dimension_results[0].ph_configuration.max_edge_length
    );
    println!(
        "  reject at alpha={:.2}: {}",
        result.alpha, result.reject_null[0]
    );
}

fn main() {
    let sphere = Sphere::new(2, 1.0).expect("failed to construct sphere");
    let sphere_x = sphere.sample(&mut ChaCha8Rng::seed_from_u64(101), SOURCE_POINTS);
    let sphere_y = sphere.sample(&mut ChaCha8Rng::seed_from_u64(102), SOURCE_POINTS);
    compare("Sphere vs. sphere", &sphere_x, &sphere_y, 1001);

    let torus = Torus::new(0.75, 0.25).expect("failed to construct torus");
    let torus_x = torus.sample(&mut ChaCha8Rng::seed_from_u64(201), SOURCE_POINTS);
    let torus_y = torus.sample(&mut ChaCha8Rng::seed_from_u64(202), SOURCE_POINTS);
    compare("Torus vs. torus", &torus_x, &torus_y, 1002);

    let klein_bottle = KleinBottle::new(0.75, 0.25).expect("failed to construct Klein bottle");
    let klein_x = klein_bottle.sample(&mut ChaCha8Rng::seed_from_u64(301), SOURCE_POINTS);
    let klein_y = klein_bottle.sample(&mut ChaCha8Rng::seed_from_u64(302), SOURCE_POINTS);
    compare("Klein bottle vs. Klein bottle", &klein_x, &klein_y, 1003);
}
