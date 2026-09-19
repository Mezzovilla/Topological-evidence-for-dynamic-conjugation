//! Serde serialization contract for `RobinsonTurnerTestResult`.
//!
//! The result type must serialize through `serde_json` (dev-dependency only;
//! the runtime API never writes files).  The `distance_matrix` field is
//! `Option<Vec<Vec<f64>>>`; depending on whether the implementation annotates
//! it with `#[serde(skip_serializing_if = "Option::is_none")]`, an absent
//! matrix serializes either as an omitted key or as an explicit `null`.
//! This test detects which convention the crate uses and pins it.

use stattda::{
    EssentialClassPolicy, InferenceMethod, PointCloud, RobinsonTurnerConfig,
    TopologicalSignatureConfig, robinson_turner_two_sample_test, topological_signature_test,
};

fn square(s: f64) -> PointCloud {
    PointCloud::try_from_rows(vec![vec![0.0, 0.0], vec![s, 0.0], vec![s, s], vec![0.0, s]]).unwrap()
}

fn tiny_groups() -> (Vec<PointCloud>, Vec<PointCloud>) {
    let a = vec![square(1.0), square(1.5)];
    let b = vec![
        PointCloud::try_from_rows(vec![vec![0.0, 0.0], vec![0.2, 0.0]]).unwrap(),
        PointCloud::try_from_rows(vec![vec![0.0, 0.0], vec![0.3, 0.0]]).unwrap(),
    ];
    (a, b)
}

fn base_config() -> RobinsonTurnerConfig {
    let mut cfg = RobinsonTurnerConfig::new(1, 4.0);
    cfg.method = InferenceMethod::Exact;
    cfg.random_seed = Some(1);
    cfg.essential_class_policy = EssentialClassPolicy::Drop;
    cfg
}

#[test]
fn result_serializes_via_serde_json() {
    let (a, b) = tiny_groups();
    let mut cfg = base_config();
    cfg.return_distance_matrix = true;
    let result = robinson_turner_two_sample_test(&a, &b, &cfg).unwrap();

    let json = serde_json::to_string(&result).expect("result must serialize");
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");

    // Spot-check the provenance contract.
    for key in [
        "statistic",
        "p_value",
        "reject_null",
        "alpha",
        "homology_dim",
        "diagram_distance",
        "ground_metric",
        "loss_q",
        "group_sizes",
        "inference_mode",
        "n_labelings_total",
        "n_random_permutations",
        "extreme_count",
        "p_value_resolution",
        "random_seed",
        "backend",
        "backend_version",
        "ph_configuration",
        "diagram_sizes",
        "dropped_essential_classes",
        "permutation_summary",
        "warnings",
        "interpretation",
        "distance_matrix",
    ] {
        assert!(
            value.get(key).is_some(),
            "serialized result is missing key `{key}`: {json}"
        );
    }
    assert!(value["distance_matrix"].is_array());
    assert_eq!(value["homology_dim"], serde_json::json!(1));
}

#[test]
fn omitted_distance_matrix_serializes_per_documented_annotation() {
    let (a, b) = tiny_groups();
    let mut cfg = base_config();
    cfg.return_distance_matrix = false;
    let result = robinson_turner_two_sample_test(&a, &b, &cfg).unwrap();
    assert!(result.distance_matrix.is_none());

    let value: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&result).unwrap()).unwrap();

    // Two acceptable Serde conventions for Option fields:
    //   * default: the key is present with value `null`;
    //   * #[serde(skip_serializing_if = "Option::is_none")]: the key is absent.
    // Pin whichever the implementation chose (see src result-type annotation;
    // documented in README).
    match value.get("distance_matrix") {
        None => { /* key omitted: skip_serializing_if convention */ }
        Some(v) => assert!(
            v.is_null(),
            "distance_matrix must be null when not returned, got {v}"
        ),
    }
}

#[test]
fn signature_result_serializes_via_serde_json() {
    let x = square(1.0);
    let y =
        PointCloud::try_from_rows(vec![vec![0.0, 0.0], vec![0.2, 0.0], vec![0.4, 0.0]]).unwrap();
    let mut cfg = TopologicalSignatureConfig::new(vec![0], 2, 4.0);
    cfg.method = InferenceMethod::Exact;
    cfg.random_seed = Some(1);
    cfg.essential_class_policy = EssentialClassPolicy::Drop;
    let result = topological_signature_test(&x, &y, &cfg).unwrap();

    let json = serde_json::to_string(&result).expect("signature result must serialize");
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");

    for key in [
        "p_values",
        "adjusted_p_values",
        "reject_null",
        "alpha",
        "homology_dimensions",
        "multiple_testing_correction",
        "random_seed",
        "sampled_group_sizes",
        "sampled_point_counts",
        "dimension_results",
        "interpretation",
    ] {
        assert!(
            value.get(key).is_some(),
            "serialized signature result is missing key `{key}`: {json}"
        );
    }
    assert_eq!(value["homology_dimensions"], serde_json::json!([0]));
    assert!(value["dimension_results"].is_array());
}
