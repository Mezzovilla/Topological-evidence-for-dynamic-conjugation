//! Backend integration tests for the Robinson–Turner two-sample test.
//!
//! The persistent-homology / diagram-distance backend (`oxicuda-tda =0.5.5`) is
//! intentionally internal, so these tests exercise it through the public API:
//! every test runs `robinson_turner_two_sample_test` with
//! `return_distance_matrix = true` and inspects the pooled
//! `distance_matrix`, `diagram_sizes`, and `dropped_essential_classes` fields of
//! the result.
//!
//! Conventions used throughout:
//! * homology degree `k = 1` tests use point clouds engineered to have known
//!   H1 diagrams (a square produces a single class born at the side length and
//!   killed at the diagonal length).
//! * homology degree `k = 0` tests use point clouds with known merge
//!   distances; every cloud then carries exactly one essential H0 class, so
//!   `EssentialClassPolicy::Drop` is required (and exercised) there.

use stattda::{
    DiagramDistance, EssentialClassPolicy, InferenceMethod, PointCloud, RobinsonTurnerConfig,
    robinson_turner_two_sample_test,
};

const SQRT2: f64 = std::f64::consts::SQRT_2;
const TOL: f64 = 1e-9;

fn cloud(rows: &[&[f64]]) -> PointCloud {
    PointCloud::try_from_rows(rows.iter().map(|r| r.to_vec()).collect())
        .expect("test cloud must be valid")
}

/// Axis-aligned square of side `s` translated by `(dx, dy)`.
/// Its Rips H1 diagram (for `s*sqrt(2) <= max_edge_length` and isolation from
/// other clusters) is exactly `{(s, s*sqrt(2))}`.
fn square(s: f64, dx: f64, dy: f64) -> PointCloud {
    cloud(&[&[dx, dy], &[dx + s, dy], &[dx + s, dy + s], &[dx, dy + s]])
}

/// Two points `d` apart: H0 diagram `{(0, d)}` plus one essential class;
/// H1 diagram empty.
fn pair(d: f64) -> PointCloud {
    cloud(&[&[0.0, 0.0], &[d, 0.0]])
}

/// Three collinear points at 0, 6, 14: H0 diagram `{(0, 6), (0, 8)}`
/// (MST edges 6 and 8) plus one essential class; H1 diagram empty.
fn triple() -> PointCloud {
    cloud(&[&[0.0, 0.0], &[6.0, 0.0], &[14.0, 0.0]])
}

fn base_config(homology_dim: usize, max_edge_length: f64) -> RobinsonTurnerConfig {
    let mut cfg = RobinsonTurnerConfig::new(homology_dim);
    cfg.max_edge_length = Some(max_edge_length);
    cfg.method = InferenceMethod::Exact;
    cfg.return_distance_matrix = true;
    cfg.random_seed = Some(7);
    cfg
}

fn distance_matrix(result: &stattda::RobinsonTurnerTestResult) -> &Vec<Vec<f64>> {
    result
        .distance_matrix
        .as_ref()
        .expect("return_distance_matrix was set")
}

// ---------------------------------------------------------------------------
// 1. Empty vs empty and empty vs nonempty diagrams give finite documented
//    distances.  A square has H1 diagram {(s, s*sqrt(2))}; a two-point cloud
//    has an empty H1 diagram.  With L-infinity ground metric and diagonal
//    matching, dist({(b,d)}, empty) = (d - b)/2 for every supported order
//    (single point), and dist(empty, empty) = 0.
// ---------------------------------------------------------------------------
#[test]
fn empty_vs_empty_and_empty_vs_nonempty_are_finite_and_documented() {
    let s = 1.0_f64;
    let group_a = [square(s, 0.0, 0.0), square(s, 0.0, 0.0)];
    let group_b = [pair(0.1), pair(0.1)];
    let cfg = base_config(1, 2.0); // sqrt(2) < 2.0 so the square's H1 dies
    let result =
        robinson_turner_two_sample_test(&group_a, &group_b, &cfg).expect("H1 has no essentials");
    let m = distance_matrix(&result);

    let expected_nonempty_vs_empty = (s * SQRT2 - s) / 2.0;
    // Order-agnostic over the pooled matrix: within-group distances are all
    // exactly 0.0 (identical nonempty diagrams in A; empty-vs-empty in B),
    // cross-group distances are all persistence/2.
    let mut zeros = 0usize;
    let mut nonempty_vs_empty = 0usize;
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if i == j {
                continue;
            }
            assert!(v.is_finite(), "distance must be finite");
            if v == 0.0 {
                zeros += 1;
            } else {
                assert!(
                    (v - expected_nonempty_vs_empty).abs() < TOL,
                    "dist({{(b,d)}}, empty) = (d-b)/2: got {v}, want {expected_nonempty_vs_empty}"
                );
                nonempty_vs_empty += 1;
            }
        }
    }
    // 4 ordered within-group pairs are exactly 0 (includes the two
    // empty-vs-empty pairs inside group B); 8 cross pairs are nonempty-vs-empty.
    assert_eq!(zeros, 4, "empty-vs-empty (and identical-diagram) distances");
    assert_eq!(nonempty_vs_empty, 8, "empty-vs-nonempty distances");
}

// ---------------------------------------------------------------------------
// 2 + 3. Dispatch evidence for W1 / W2 / Bottleneck, and a hand-specified
//    cloud whose diagram distances match direct expectations.
//
//    One cloud containing two well-separated squares (sides 1 and 2) has H1
//    diagram D = {(1, sqrt(2)), (2, 2*sqrt(2))}.  Against the empty diagram
//    with the L-infinity ground metric:
//      W1(D, 0)        = (p1 + p2) / 2
//      W2(D, 0)        = sqrt(p1^2 + p2^2) / 2
//      Bottleneck(D,0) = max(p1, p2) / 2 = p2 / 2
//    where p1 = sqrt(2)-1, p2 = 2*(sqrt(2)-1).  These three values are all
//    distinct, so observing each expected value under its configured
//    `DiagramDistance` proves the order dispatches to the intended backend
//    function.
// ---------------------------------------------------------------------------
fn two_square_cloud() -> PointCloud {
    cloud(&[
        &[0.0, 0.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
        &[0.0, 1.0],
        &[10.0, 10.0],
        &[12.0, 10.0],
        &[12.0, 12.0],
        &[10.0, 12.0],
    ])
}

#[test]
fn diagram_distance_orders_dispatch_to_expected_metrics() {
    let p1 = SQRT2 - 1.0;
    let p2 = 2.0 * (SQRT2 - 1.0);
    let expected = [
        (DiagramDistance::Wasserstein1, (p1 + p2) / 2.0),
        (
            DiagramDistance::Wasserstein2,
            (p1 * p1 + p2 * p2).sqrt() / 2.0,
        ),
        (DiagramDistance::Bottleneck, p2 / 2.0),
    ];
    // Sanity: the three expectations must be mutually distinct for this to be
    // dispatch evidence.
    assert!(expected[0].1 != expected[1].1 && expected[1].1 != expected[2].1);

    for (order, want) in expected {
        let group_a = [two_square_cloud(), two_square_cloud()];
        let group_b = [pair(0.1), pair(0.1)];
        let mut cfg = base_config(1, 3.0); // 2*sqrt(2) < 3.0 << cluster gap
        cfg.diagram_distance = order;
        let result =
            robinson_turner_two_sample_test(&group_a, &group_b, &cfg).expect("test must succeed");
        let m = distance_matrix(&result);
        // Cross-group distances are all D-vs-empty; within-group are all 0.
        let got = m
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter(move |(j, _)| *j != i)
                    .map(|(_, &v)| v)
            })
            .fold(0.0_f64, f64::max);
        assert!(
            (got - want).abs() < TOL,
            "{order:?}: got {got}, want {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Adversarial pair where greedy matching differs from optimal.
//
//    Work in H0 (Drop policy removes the single essential class per cloud).
//      cloud X (triple): DX = {(0,6), (0,8)}
//      cloud Y (pair 6.9): DY = {(0,6.9)}
//    L-infinity costs:  x1=(0,6): diag 3, to y1 = 0.9
//                       x2=(0,8): diag 4, to y1 = 1.1
//    A greedy matcher takes the cheapest edge x1-y1 (0.9) first and is then
//    forced into x2-diagonal (4): bottleneck 4, W1 4.9.
//    The optimal matching pairs x2-y1 (1.1) and x1-diagonal (3):
//      Bottleneck = 3, W1 = 4.1, W2 = sqrt(1.1^2 + 3^2) = sqrt(10.21).
// ---------------------------------------------------------------------------
#[test]
fn adversarial_diagram_pair_matches_optimal_not_greedy() {
    let cases = [
        (DiagramDistance::Bottleneck, 3.0_f64),
        (DiagramDistance::Wasserstein1, 4.1_f64),
        (DiagramDistance::Wasserstein2, (10.21_f64).sqrt()),
    ];
    for (order, want) in cases {
        let group_a = [triple(), triple()];
        let group_b = [pair(6.9), pair(6.9)];
        let mut cfg = base_config(0, 10.0);
        cfg.diagram_distance = order;
        cfg.essential_class_policy = EssentialClassPolicy::Drop;
        let result =
            robinson_turner_two_sample_test(&group_a, &group_b, &cfg).expect("test must succeed");
        let m = distance_matrix(&result);
        // Order-agnostic: every cross-group distance equals the optimal value;
        // within-group distances are 0, so the off-diagonal max suffices.
        let got = m
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter(move |(j, _)| *j != i)
                    .map(|(_, &v)| v)
            })
            .fold(0.0_f64, f64::max);
        assert!(
            (got - want).abs() < 1e-6,
            "{order:?}: optimal is {want}; a greedy matcher would return a \
             larger value (e.g. bottleneck 4.0 / W1 4.9); got {got}"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Each cloud produces exactly one selected-degree diagram; no pooling of
//    raw points across clouds.  diagram_sizes must have one entry per cloud
//    (N total) and be reproducible across identical calls.
// ---------------------------------------------------------------------------
#[test]
fn one_diagram_per_cloud_no_pooling() {
    let group_a = [square(1.0, 0.0, 0.0), square(1.0, 0.0, 0.0)];
    let group_b = [pair(0.1), pair(0.1)];
    let cfg = base_config(1, 2.0);
    let r1 = robinson_turner_two_sample_test(&group_a, &group_b, &cfg).unwrap();
    let r2 = robinson_turner_two_sample_test(&group_a, &group_b, &cfg).unwrap();

    assert_eq!(r1.diagram_sizes.len(), 4, "one diagram per cloud");
    assert_eq!(
        r1.diagram_sizes, r2.diagram_sizes,
        "per-cloud reproducibility"
    );
    // Two clouds carry a single H1 class, two carry none — regardless of
    // internal ordering this proves diagrams are per-cloud, not pooled.
    let ones = r1.diagram_sizes.iter().filter(|&&s| s == 1).count();
    let zeros = r1.diagram_sizes.iter().filter(|&&s| s == 0).count();
    assert_eq!((ones, zeros), (2, 2));
}

// ---------------------------------------------------------------------------
// 6. Essential classes surface before policy handling: H0 clouds always have
//    one essential class, so Reject must error and Drop must record counts.
// ---------------------------------------------------------------------------
#[test]
fn essential_classes_reject_then_drop_records_counts() {
    let group_a = [triple(), triple()];
    let group_b = [pair(6.9), pair(6.9)];

    let mut cfg = base_config(0, 10.0);
    cfg.essential_class_policy = EssentialClassPolicy::Reject;
    let err = robinson_turner_two_sample_test(&group_a, &group_b, &cfg);
    assert!(
        err.is_err(),
        "Reject policy must surface essential H0 classes as an error"
    );

    cfg.essential_class_policy = EssentialClassPolicy::Drop;
    let result = robinson_turner_two_sample_test(&group_a, &group_b, &cfg)
        .expect("Drop policy must censor and record");
    assert_eq!(result.dropped_essential_classes.len(), 4);
    assert!(
        result.dropped_essential_classes.iter().all(|&c| c == 1),
        "each cloud has exactly one essential H0 class: {:?}",
        result.dropped_essential_classes
    );
}

// ---------------------------------------------------------------------------
// 7. Distance matrix is symmetric with an exact zero diagonal, and each
//    unordered pair is computed exactly once (doc-hidden test seam).
// ---------------------------------------------------------------------------
#[test]
fn distance_matrix_symmetric_zero_diagonal_pairs_computed_once() {
    stattda::__test_seams::reset_pairwise_distance_calls();

    // Four distinct clouds -> N = 4 -> exactly C(4,2) = 6 distance calls.
    let group_a = [square(1.0, 0.0, 0.0), square(1.5, 0.0, 0.0)];
    let group_b = [pair(0.1), two_square_cloud()];
    let cfg = base_config(1, 3.0);
    let result = robinson_turner_two_sample_test(&group_a, &group_b, &cfg).unwrap();
    let m = distance_matrix(&result);
    let n = m.len();
    assert_eq!(n, 4);
    for (i, row) in m.iter().enumerate() {
        assert_eq!(row[i], 0.0, "diagonal must be exactly zero");
        for (j, &v) in row.iter().enumerate() {
            assert!(
                (v - m[j][i]).abs() == 0.0 || (v - m[j][i]).abs() < TOL,
                "matrix must be symmetric"
            );
        }
    }
    assert_eq!(
        stattda::__test_seams::pairwise_distance_calls(),
        n * (n - 1) / 2,
        "each unordered pair must be computed exactly once"
    );
}
