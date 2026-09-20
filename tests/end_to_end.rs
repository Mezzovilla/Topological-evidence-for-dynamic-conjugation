//! Deterministic end-to-end tests for `robinson_turner_two_sample_test`.
//!
//! Datasets are kept deliberately modest (the pinned backend enumerates Rips
//! simplices, so clouds stay at <= 16 points and `homology_dim = 1`).  All
//! randomness uses a local splitmix64 generator so the tests are fully
//! deterministic and need no RNG dev-dependency.

use stattda::{
    EssentialClassPolicy, InferenceMethod, PointCloud, RobinsonTurnerConfig,
    robinson_turner_two_sample_test,
};

/// Minimal deterministic RNG (splitmix64). Fixed seeds => fixed datasets.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Roughly symmetric noise in [-scale, scale] (sum of two uniforms).
    fn noise(&mut self, scale: f64) -> f64 {
        (self.f64() + self.f64() - 1.0) * scale
    }
}

fn cloud(rows: Vec<Vec<f64>>) -> PointCloud {
    PointCloud::try_from_rows(rows).expect("generated cloud must be valid")
}

/// `n` points on a circle of `radius`, angles evenly spaced with jitter,
/// radial noise `noise`.
fn noisy_circle(rng: &mut Rng, n: usize, radius: f64, noise: f64) -> PointCloud {
    let rows = (0..n)
        .map(|i| {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64) + rng.noise(0.15);
            let r = radius + rng.noise(noise);
            vec![r * theta.cos(), r * theta.sin()]
        })
        .collect();
    cloud(rows)
}

/// Two concentric noisy circles in one cloud (radii 1 and 2).
fn noisy_two_circles(rng: &mut Rng, n_each: usize, noise: f64) -> PointCloud {
    let mut rows = Vec::with_capacity(2 * n_each);
    for &radius in &[1.0_f64, 2.0] {
        for i in 0..n_each {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n_each as f64) + rng.noise(0.15);
            let r = radius + rng.noise(noise);
            rows.push(vec![r * theta.cos(), r * theta.sin()]);
        }
    }
    cloud(rows)
}

fn mc_config(seed: u64, permutations: u64) -> RobinsonTurnerConfig {
    let mut cfg = RobinsonTurnerConfig::new(1);
    cfg.max_edge_length = Some(4.0);
    cfg.method = InferenceMethod::MonteCarlo;
    cfg.n_permutations = permutations;
    cfg.random_seed = Some(seed);
    cfg.essential_class_policy = EssentialClassPolicy::Drop; // inert for H1
    cfg
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// ---------------------------------------------------------------------------
// Null smoke test: clouds drawn from ONE noisy-circle process, split into two
// arbitrary groups. Under the null the lower-tail p-value should be roughly
// uniform; the rejection fraction at alpha = 0.20 must be broadly compatible
// with the nominal level.
//
// Prespecified non-flaky tolerance: with REPS = 15 repetitions and nominal
// alpha = 0.20 we expect ~3 rejections; we require <= 6 (i.e. fraction <= 0.40,
// twice the nominal level). Binomial(15, 0.2) gives P(X >= 7) < 0.02 for a
// truly uniform p, and seeds are fixed anyway so the test is deterministic.
// ---------------------------------------------------------------------------
#[test]
fn null_smoke_rejection_fraction_compatible_with_alpha() {
    const REPS: usize = 15;
    const ALPHA: f64 = 0.20;
    let mut rejections = 0usize;
    for rep in 0..REPS {
        let mut rng = Rng::new(0xC1EC_1E00 + rep as u64);
        // One process, eight clouds, split 4/4.
        let all: Vec<PointCloud> = (0..8)
            .map(|_| noisy_circle(&mut rng, 10, 1.0, 0.08))
            .collect();
        let mut cfg = mc_config(0xBEEF_0000 + rep as u64, 199);
        cfg.alpha = ALPHA;
        let result = robinson_turner_two_sample_test(&all[..4], &all[4..], &cfg)
            .expect("null test must succeed");
        assert!(result.p_value > 0.0, "MC p-value is never zero");
        if result.reject_null {
            rejections += 1;
        }
    }
    assert!(
        rejections * 5 <= REPS * 2,
        "rejection fraction {rejections}/{REPS} exceeds prespecified 2x nominal tolerance at alpha {ALPHA}"
    );
}

// ---------------------------------------------------------------------------
// Alternative sensitivity: noisy one-circle clouds vs noisy two-concentric-
// circle clouds, homology_dim = 1. The two-circle clouds carry two prominent
// H1 classes, so the median alternative p-value must sit below the median
// null p-value for the same seeds.
// ---------------------------------------------------------------------------
#[test]
fn alternative_sensitivity_median_p_below_null() {
    const REPS: usize = 7;
    let mut null_p = Vec::with_capacity(REPS);
    let mut alt_p = Vec::with_capacity(REPS);
    for rep in 0..REPS {
        let data_seed = 0xDA7A_0000 + rep as u64;
        let perm_seed = 0x9EED_0000 + rep as u64;

        // Null arm: one process split into two groups.
        let mut rng = Rng::new(data_seed);
        let all: Vec<PointCloud> = (0..8)
            .map(|_| noisy_circle(&mut rng, 10, 1.0, 0.08))
            .collect();
        let cfg = mc_config(perm_seed, 199);
        null_p.push(
            robinson_turner_two_sample_test(&all[..4], &all[4..], &cfg)
                .unwrap()
                .p_value,
        );

        // Alternative arm: circles vs concentric circle pairs.
        let mut rng = Rng::new(data_seed); // same seed, same draw sequence
        let a: Vec<PointCloud> = (0..4)
            .map(|_| noisy_circle(&mut rng, 10, 1.0, 0.08))
            .collect();
        let b: Vec<PointCloud> = (0..4)
            .map(|_| noisy_two_circles(&mut rng, 8, 0.08))
            .collect();
        alt_p.push(
            robinson_turner_two_sample_test(&a, &b, &cfg)
                .unwrap()
                .p_value,
        );
    }
    let (mn, ma) = (median(&mut null_p), median(&mut alt_p));
    assert!(
        ma < mn,
        "median alternative p ({ma}) must be below median null p ({mn})"
    );
}

// ---------------------------------------------------------------------------
// Affine-scale behavior: the statistic is NOT scale invariant. Uniformly
// scaling every cloud by `c` scales every Rips filtration value (and hence
// every diagram birth/death and diagram distance) by `c`; with the default
// loss exponent q = 2 the observed statistic therefore scales by c^2.
// We keep max_edge_length scaled in lockstep so the diagrams themselves are
// identical up to the scale factor — isolating the metric's scale dependence.
// ---------------------------------------------------------------------------
#[test]
fn uniform_scaling_changes_statistic_not_scale_invariant() {
    let group_a = [
        cloud(vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ]),
        cloud(vec![
            vec![0.0, 0.0],
            vec![1.5, 0.0],
            vec![1.5, 1.5],
            vec![0.0, 1.5],
        ]),
    ];
    let group_b = [
        cloud(vec![vec![0.0, 0.0], vec![0.2, 0.0]]),
        cloud(vec![vec![0.0, 0.0], vec![0.3, 0.0]]),
    ];
    let scale = 2.0_f64;
    let scaled_a: Vec<PointCloud> = group_a
        .iter()
        .map(|c| PointCloud {
            coordinates: c.coordinates.iter().map(|x| x * scale).collect(),
            n_points: c.n_points,
            ambient_dim: c.ambient_dim,
        })
        .collect();
    let scaled_b: Vec<PointCloud> = group_b
        .iter()
        .map(|c| PointCloud {
            coordinates: c.coordinates.iter().map(|x| x * scale).collect(),
            n_points: c.n_points,
            ambient_dim: c.ambient_dim,
        })
        .collect();

    let mut cfg = RobinsonTurnerConfig::new(1);
    cfg.max_edge_length = Some(4.0);
    cfg.method = InferenceMethod::Exact;
    let r1 = robinson_turner_two_sample_test(&group_a, &group_b, &cfg).unwrap();
    cfg.max_edge_length = Some(4.0 * scale);
    let r2 = robinson_turner_two_sample_test(&scaled_a, &scaled_b, &cfg).unwrap();

    assert!(r1.statistic > 0.0);
    // Distances scale by `scale`; with q = 2 the statistic scales by scale^2.
    let expected = r1.statistic * scale * scale;
    assert!(
        (r2.statistic - expected).abs() <= expected * 1e-6,
        "scaling data by {scale} must scale the F-statistic by {}: got {}, want {}",
        scale * scale,
        r2.statistic,
        expected
    );
    assert_ne!(
        r1.statistic, r2.statistic,
        "documented: the default analysis is not scale invariant"
    );
}
