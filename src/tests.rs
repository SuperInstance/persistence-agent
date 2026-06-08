use crate::agent_features::{AgentArchetype, AgentProfile, AgentProfiler};
use crate::barcode::{Barcode, PersistencePair};
use crate::boundary::BoundaryMatrix;
use crate::error::PersistenceError;
use crate::point_cloud::PointCloud;
use crate::vietoris_rips::VietorisRipsComplex;
use std::f64::consts::PI;

// ── PointCloud ──────────────────────────────────────────────

#[test]
fn test_point_cloud_empty() {
    let result = PointCloud::new(vec![]);
    assert!(matches!(result, Err(PersistenceError::EmptyCloud)));
}

#[test]
fn test_point_cloud_single_point() {
    let cloud = PointCloud::new(vec![vec![1.0, 2.0]]).unwrap();
    assert_eq!(cloud.n_points(), 1);
    assert_eq!(cloud.dimension(), 2);
    assert_eq!(cloud.distance_matrix[0][0], 0.0);
}

#[test]
fn test_point_cloud_distance_matrix() {
    let cloud = PointCloud::new(vec![vec![0.0], vec![3.0], vec![5.0]]).unwrap();
    assert!((cloud.distance(0, 1) - 3.0).abs() < 1e-10);
    assert!((cloud.distance(1, 2) - 2.0).abs() < 1e-10);
    assert!((cloud.distance(0, 2) - 5.0).abs() < 1e-10);
    assert!((cloud.distance(2, 0) - 5.0).abs() < 1e-10);
}

#[test]
fn test_point_cloud_dimension_mismatch() {
    let result = PointCloud::new(vec![vec![0.0, 0.0], vec![1.0]]);
    assert!(matches!(
        result,
        Err(PersistenceError::DimensionMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn test_knn_basic() {
    let cloud = PointCloud::new(vec![vec![0.0], vec![1.0], vec![5.0]]).unwrap();
    let knn = cloud.knn(1).unwrap();
    assert_eq!(knn[0], vec![1]);
    assert_eq!(knn[1], vec![0]);
    assert_eq!(knn[2], vec![1]);
}

#[test]
fn test_knn_invalid_k() {
    let cloud = PointCloud::new(vec![vec![0.0], vec![1.0]]).unwrap();
    assert!(matches!(cloud.knn(0), Err(PersistenceError::InvalidK { .. })));
    assert!(matches!(cloud.knn(2), Err(PersistenceError::InvalidK { .. })));
}

#[test]
fn test_unique_distances() {
    let cloud = PointCloud::new(vec![vec![0.0], vec![1.0], vec![3.0]]).unwrap();
    let dists = cloud.unique_distances();
    assert_eq!(dists.len(), 3);
    assert!((dists[0] - 1.0).abs() < 1e-10);
    assert!((dists[1] - 2.0).abs() < 1e-10);
    assert!((dists[2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_max_distance() {
    let cloud = PointCloud::new(vec![vec![0.0], vec![5.0]]).unwrap();
    assert!((cloud.max_distance() - 5.0).abs() < 1e-10);
}

#[test]
fn test_point_cloud_clone() {
    let cloud = PointCloud::new(vec![vec![0.0, 1.0], vec![2.0, 3.0]]).unwrap();
    let cloned = cloud.clone();
    assert_eq!(cloud.n_points(), cloned.n_points());
    assert!((cloud.distance(0, 1) - cloned.distance(0, 1)).abs() < 1e-10);
}

// ── VietorisRipsComplex ─────────────────────────────────────

#[test]
fn test_vr_triangle() {
    let cloud = PointCloud::new(vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.5, 0.866],
    ]).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 2, f64::INFINITY).unwrap();
    assert_eq!(vr.n_simplices(), 7);
    assert_eq!(vr.simplices_of_dimension(0).len(), 3);
    assert_eq!(vr.simplices_of_dimension(1).len(), 3);
    assert_eq!(vr.simplices_of_dimension(2).len(), 1);
}

#[test]
fn test_vr_max_eps_pruning() {
    let cloud = PointCloud::new(vec![vec![0.0], vec![1.0], vec![10.0]]).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 1, 2.0).unwrap();
    assert_eq!(vr.simplices_of_dimension(1).len(), 1);
}

#[test]
fn test_vr_simplex_dimension() {
    let cloud = PointCloud::new(vec![vec![0.0], vec![1.0], vec![2.0]]).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 2, f64::INFINITY).unwrap();
    assert_eq!(vr.simplex_dimension(0), 0);
    assert_eq!(vr.simplex_dimension(3), 1);
}

// ── BoundaryMatrix ──────────────────────────────────────────

#[test]
fn test_boundary_triangle_reduce() {
    let cloud = PointCloud::new(vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.5, 0.866],
    ]).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 2, f64::INFINITY).unwrap();
    let mut bm = BoundaryMatrix::build(&vr).unwrap();
    let low_map = bm.reduce();
    assert!(!low_map.is_empty());
}

#[test]
fn test_boundary_zero_columns_for_vertices() {
    let cloud = PointCloud::new(vec![vec![0.0], vec![1.0], vec![2.0]]).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 1, f64::INFINITY).unwrap();
    let bm = BoundaryMatrix::build(&vr).unwrap();
    for i in 0..3 {
        assert!(bm.matrix[i].iter().all(|&v| v == 0));
    }
}

// ── Barcode: known shapes ───────────────────────────────────

fn circle_points(n: usize, radius: f64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let theta = 2.0 * PI * i as f64 / n as f64;
            vec![radius * theta.cos(), radius * theta.sin()]
        })
        .collect()
}

#[test]
fn test_barcode_circle_betti1() {
    let cloud = PointCloud::new(circle_points(8, 1.0)).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 2, f64::INFINITY).unwrap();
    let barcode = Barcode::compute(&vr).unwrap();
    let h1 = barcode.pairs_of_dimension(1);
    assert!(!h1.is_empty(), "Circle should produce at least one H1 feature");
}

#[test]
fn test_barcode_circle_betti0() {
    let cloud = PointCloud::new(circle_points(8, 1.0)).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 2, f64::INFINITY).unwrap();
    let barcode = Barcode::compute(&vr).unwrap();
    let h0 = barcode.pairs_of_dimension(0);
    let infinite_h0 = h0.iter().filter(|p| !p.death.is_finite()).count();
    assert!(infinite_h0 >= 1, "Circle should have ≥1 infinite H0 feature");
}

#[test]
fn test_figure_eight_betti1() {
    let mut pts = Vec::new();
    for i in 0..6 {
        let theta = 2.0 * PI * i as f64 / 6.0;
        pts.push(vec![-1.0 + 0.5 * theta.cos(), 0.5 * theta.sin()]);
    }
    for i in 0..6 {
        let theta = 2.0 * PI * i as f64 / 6.0;
        pts.push(vec![1.0 + 0.5 * theta.cos(), 0.5 * theta.sin()]);
    }
    let cloud = PointCloud::new(pts).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 2, f64::INFINITY).unwrap();
    let barcode = Barcode::compute(&vr).unwrap();
    let h1 = barcode.pairs_of_dimension(1);
    assert!(h1.len() >= 2, "Figure-eight should have ≥2 H1 features, got {}", h1.len());
}

#[test]
fn test_two_clusters_betti0() {
    let mut pts = Vec::new();
    for i in 0..5 { pts.push(vec![i as f64 * 0.1, 0.0]); }
    for i in 0..5 { pts.push(vec![10.0 + i as f64 * 0.1, 0.0]); }
    let cloud = PointCloud::new(pts).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 1, 5.0).unwrap();
    let barcode = Barcode::compute(&vr).unwrap();
    let betti_at_1 = barcode.betti_numbers_at(1.0);
    assert!(betti_at_1[0] >= 2, "Two clusters should have β₀ ≥ 2 at intermediate epsilon");
}

#[test]
fn test_single_point_barcode() {
    let cloud = PointCloud::new(vec![vec![0.0, 0.0]]).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 1, f64::INFINITY).unwrap();
    let barcode = Barcode::compute(&vr).unwrap();
    let h0 = barcode.pairs_of_dimension(0);
    assert_eq!(h0.len(), 1);
    assert!(!h0[0].death.is_finite());
}

#[test]
fn test_betti_curve_sorted() {
    let cloud = PointCloud::new(circle_points(6, 1.0)).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 2, f64::INFINITY).unwrap();
    let barcode = Barcode::compute(&vr).unwrap();
    assert!(!barcode.betti_curve.is_empty());
    for w in barcode.betti_curve.windows(2) {
        assert!(w[0].0 <= w[1].0);
    }
}

#[test]
fn test_persistence_pair_value() {
    let p = PersistencePair { birth: 1.0, death: 3.5, dimension: 0 };
    assert!((p.persistence() - 2.5).abs() < 1e-10);
}

#[test]
fn test_barcode_entropy_single_point() {
    let cloud = PointCloud::new(vec![vec![0.0]]).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 1, f64::INFINITY).unwrap();
    let barcode = Barcode::compute(&vr).unwrap();
    assert!((barcode.persistence_entropy() - 0.0).abs() < 1e-10);
}

#[test]
fn test_barcode_serialization() {
    let cloud = PointCloud::new(circle_points(4, 1.0)).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 1, f64::INFINITY).unwrap();
    let barcode = Barcode::compute(&vr).unwrap();
    let json = serde_json::to_string(&barcode).unwrap();
    assert!(json.contains("birth"));
    let back: Barcode = serde_json::from_str(&json).unwrap();
    assert_eq!(barcode.pairs.len(), back.pairs.len());
}

// ── AgentProfiler ───────────────────────────────────────────

#[test]
fn test_steady_agent() {
    let mut pts = Vec::new();
    for i in 0..10 {
        pts.push(vec![0.5 + i as f64 * 0.01, 0.5 + (i as f64 * 0.01).sin()]);
    }
    let profiler = AgentProfiler::new(1);
    let profile = profiler.profile(pts).unwrap();
    assert!(
        profile.archetype == AgentArchetype::Steady || profile.archetype == AgentArchetype::Balanced,
        "Tight cluster should be Steady or Balanced, got {:?}", profile.archetype
    );
}

#[test]
fn test_volatile_agent() {
    let pts: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 100.0, i as f64 * 100.0]).collect();
    let profiler = AgentProfiler::new(1);
    let profile = profiler.profile(pts).unwrap();
    assert!(profile.persistence_entropy >= 0.0);
}

#[test]
fn test_profile_has_betti_numbers() {
    let pts = circle_points(6, 1.0);
    let profiler = AgentProfiler::new(1);
    let profile = profiler.profile(pts).unwrap();
    assert!(!profile.betti_numbers.is_empty());
}

#[test]
fn test_profile_max_persistence() {
    let pts = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let profiler = AgentProfiler::new(1);
    let profile = profiler.profile(pts).unwrap();
    assert!(profile.max_persistence >= 0.0);
}

#[test]
fn test_agent_profile_serialization() {
    let pts = circle_points(6, 1.0);
    let profiler = AgentProfiler::new(1);
    let profile = profiler.profile(pts).unwrap();
    let json = serde_json::to_string(&profile).unwrap();
    assert!(json.contains("archetype"));
    let back: AgentProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(profile.archetype, back.archetype);
}

#[test]
fn test_archetype_display() {
    assert_eq!(format!("{}", AgentArchetype::Steady), "Steady");
    assert_eq!(format!("{}", AgentArchetype::Explorer), "Explorer");
    assert_eq!(format!("{}", AgentArchetype::Volatile), "Volatile");
    assert_eq!(format!("{}", AgentArchetype::Deep), "Deep");
    assert_eq!(format!("{}", AgentArchetype::Balanced), "Balanced");
}

#[test]
fn test_collinear_points() {
    let cloud = PointCloud::new(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 1, f64::INFINITY).unwrap();
    let barcode = Barcode::compute(&vr).unwrap();
    let h0 = barcode.pairs_of_dimension(0);
    assert!(!h0.is_empty());
}

#[test]
fn test_knn_k2() {
    let cloud = PointCloud::new(vec![vec![0.0], vec![1.0], vec![4.0], vec![5.0]]).unwrap();
    let knn = cloud.knn(2).unwrap();
    assert_eq!(knn.len(), 4);
    assert_eq!(knn[0], vec![1, 2]);
}

#[test]
fn test_barcode_max_persistence() {
    let cloud = PointCloud::new(vec![vec![0.0], vec![1.0]]).unwrap();
    let vr = VietorisRipsComplex::build(&cloud, 1, f64::INFINITY).unwrap();
    let barcode = Barcode::compute(&vr).unwrap();
    assert!(barcode.max_persistence() >= 0.0);
}
