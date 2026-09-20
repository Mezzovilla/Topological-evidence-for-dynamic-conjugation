//! Compare topological signatures between point clouds sampled from
//! distinct manifolds (sphere, torus, Klein bottle) using exact inference.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use stattda::datasets::{KleinBottle, PointCloudGenerator, Sphere, Torus};
use stattda::{
    EssentialClassPolicy, InferenceMethod, PointCloud, TopologicalSignatureConfig,
    topological_signature_test,
};

const SOURCE_POINTS: usize = 100;
const SAMPLED_POINTS: usize = 40;
const N_SAMPLES: usize = 6;

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
    let torus = Torus::new(0.75, 0.25).expect("failed to construct torus");
    let klein = KleinBottle::new(0.75, 0.25).expect("failed to construct Klein bottle");

    let mut sphere_rng = ChaCha8Rng::seed_from_u64(401);
    let sphere_cloud = sphere.sample(&mut sphere_rng, SOURCE_POINTS);

    let mut torus_rng = ChaCha8Rng::seed_from_u64(402);
    let torus_cloud = torus.sample(&mut torus_rng, SOURCE_POINTS);

    let mut klein_rng = ChaCha8Rng::seed_from_u64(403);
    let klein_cloud = klein.sample(&mut klein_rng, SOURCE_POINTS);

    compare("Sphere vs. torus", &sphere_cloud, &torus_cloud, 2001);
    compare("Sphere vs. Klein bottle", &sphere_cloud, &klein_cloud, 2002);
    compare("Torus vs. Klein bottle", &torus_cloud, &klein_cloud, 2003);
}
