pub mod point_cloud;
pub mod vietoris_rips;
pub mod boundary;
pub mod reduction;
pub mod barcode;
pub mod agent_features;

pub use point_cloud::{ActionPoint, Metric, PointCloud};
pub use vietoris_rips::VRComplex;
pub use boundary::BoundaryMatrix;
pub use barcode::{Barcode, BarcodeCollection};
pub use agent_features::AgentFeatures;

use reduction::{reduce, pairs_to_barcodes};

/// Run the full persistence pipeline: actions → point cloud → VR → boundary → reduce → barcodes → features.
pub fn analyze(
    points: Vec<ActionPoint>,
    metric: Metric,
    max_dim: usize,
) -> (BarcodeCollection, AgentFeatures) {
    let cloud = PointCloud::new(points, metric);
    let vr = VRComplex::build_filtration(&cloud, max_dim);
    let bm = BoundaryMatrix::from_vr_complex(&vr);
    let result = reduce(&bm);
    let barcodes = pairs_to_barcodes(&result, &vr, max_dim);
    let collection = BarcodeCollection::new(barcodes);
    let features = AgentFeatures::from_barcodes(&collection);
    (collection, features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_pipeline_single_point() {
        let points = vec![ActionPoint::new("agent", 0.0, vec![1.0, 0.0, 0.0])];
        let (barcodes, features) = analyze(points, Metric::Euclidean, 1);
        let h0 = barcodes.dimension(0).unwrap();
        assert_eq!(h0.bars.len(), 1);
        assert!(h0.bars[0].1.is_infinite());
        assert!((features.stability - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_full_pipeline_two_points() {
        let points = vec![
            ActionPoint::new("agent", 0.0, vec![0.0]),
            ActionPoint::new("agent", 1.0, vec![2.0]),
        ];
        let (barcodes, _features) = analyze(points, Metric::Euclidean, 1);
        let h0 = barcodes.dimension(0).unwrap();
        // Two points: one [0,∞) and one [0, 2.0)
        assert_eq!(h0.bars.len(), 2);
        let inf_count = h0.bars.iter().filter(|b| b.1.is_infinite()).count();
        assert_eq!(inf_count, 1);
    }

    #[test]
    fn test_full_pipeline_three_points_triangle() {
        let points = vec![
            ActionPoint::new("agent", 0.0, vec![0.0, 0.0]),
            ActionPoint::new("agent", 1.0, vec![1.0, 0.0]),
            ActionPoint::new("agent", 2.0, vec![0.5, 0.866]),
        ];
        let (barcodes, features) = analyze(points, Metric::Euclidean, 2);
        let h0 = barcodes.dimension(0).unwrap();
        assert!(h0.bars.len() >= 1);
        assert_eq!(features.personality_vector.len(), 3);
    }

    #[test]
    fn test_full_pipeline_cosine_metric() {
        let points = vec![
            ActionPoint::new("agent", 0.0, vec![1.0, 0.0]),
            ActionPoint::new("agent", 1.0, vec![0.0, 1.0]),
            ActionPoint::new("agent", 2.0, vec![1.0, 1.0]),
        ];
        let (barcodes, _features) = analyze(points, Metric::Cosine, 2);
        assert!(barcodes.dimension(0).is_some());
    }

    #[test]
    fn test_serde_roundtrip_action_point() {
        let ap = ActionPoint::new("test-agent", 1234567890.0, vec![1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&ap).unwrap();
        let decoded: ActionPoint = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.agent_id, "test-agent");
        assert!((decoded.timestamp - 1234567890.0).abs() < 1e-9);
        assert_eq!(decoded.features, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_serde_roundtrip_point_cloud() {
        let pc = PointCloud::new(
            vec![ActionPoint::new("a", 0.0, vec![1.0, 2.0])],
            Metric::Cosine,
        );
        let json = serde_json::to_string(&pc).unwrap();
        let decoded: PointCloud = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.metric, Metric::Cosine);
        assert_eq!(decoded.points.len(), 1);
    }

    #[test]
    fn test_serde_roundtrip_vr_complex() {
        let vr = VRComplex {
            simplices: vec![vec![0], vec![1], vec![0, 1]],
            filtration_values: vec![0.0, 0.0, 1.5],
        };
        let json = serde_json::to_string(&vr).unwrap();
        let decoded: VRComplex = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.simplices, vr.simplices);
        assert_eq!(decoded.filtration_values, vr.filtration_values);
    }

    #[test]
    fn test_serde_roundtrip_boundary_matrix() {
        let bm = BoundaryMatrix {
            matrix: vec![vec![0, 0], vec![1, 1]],
            dim: 1,
            n_cols: 2,
            n_rows: 2,
        };
        let json = serde_json::to_string(&bm).unwrap();
        let decoded: BoundaryMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.matrix, bm.matrix);
    }

    #[test]
    fn test_serde_roundtrip_barcode() {
        let barcode = Barcode {
            dimension: 1,
            bars: vec![(0.5, 2.0), (1.0, 3.0)],
        };
        let json = serde_json::to_string(&barcode).unwrap();
        let decoded: Barcode = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.dimension, 1);
        assert_eq!(decoded.bars.len(), 2);
        assert!((decoded.bars[0].0 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_barcode_infinity_bar() {
        // Infinity doesn't survive JSON serde (it becomes null),
        // so we test the logic directly.
        let barcode = Barcode {
            dimension: 0,
            bars: vec![(0.0, f64::INFINITY)],
        };
        assert!(barcode.bars[0].1.is_infinite());
        assert_eq!(barcode.betti_at(999.0), 1);
    }

    #[test]
    fn test_serde_roundtrip_barcode_collection() {
        let collection = BarcodeCollection::new(vec![
            Barcode { dimension: 0, bars: vec![(0.0, 2.5)] },
        ]);
        let json = serde_json::to_string(&collection).unwrap();
        let decoded: BarcodeCollection = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.barcodes.len(), 1);
    }

    #[test]
    fn test_pipeline_many_actions() {
        // 10 points in 2D — agent with varied behavior
        let points: Vec<ActionPoint> = (0..10)
            .map(|i| ActionPoint::new("agent", i as f64, vec![i as f64, (i as f64).sin()]))
            .collect();
        let (barcodes, features) = analyze(points, Metric::Euclidean, 2);
        assert!(barcodes.dimension(0).is_some());
        assert_eq!(features.personality_vector.len(), 3);
    }
}
