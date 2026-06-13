use crate::boundary::BoundaryMatrix;
use crate::vietoris_rips::VRComplex;
use crate::barcode::Barcode;

/// Result of the reduction algorithm: reduced matrix + birth-death pairs.
#[derive(Debug, Clone)]
pub struct ReductionResult {
    pub reduced: Vec<Vec<i32>>,
    /// (birth_simplex_idx, death_simplex_idx) pairs
    pub pairs: Vec<(usize, Option<usize>)>,
}

/// Perform standard column reduction on the boundary matrix over Z₂.
///
/// Returns birth-death pairs indexed by simplex position.
/// A pair `(birth, Some(death))` means a feature born at simplex `birth`
/// and died at simplex `death`. `(birth, None)` means the feature persists.
pub fn reduce(bm: &BoundaryMatrix) -> ReductionResult {
    let n = bm.n_cols;
    let mut reduced = bm.matrix.clone();

    // Track which column has a given low value
    let mut low_to_col: Vec<Option<usize>> = vec![None; n];

    for j in 0..n {
        // Reduce column j
        let mut low_j = find_low(&reduced, j);
        while let Some(low) = low_j {
            if let Some(k) = low_to_col[low] {
                // Add column k to column j (mod 2)
                for i in 0..n {
                    reduced[j][i] = (reduced[j][i] + reduced[k][i]) % 2;
                }
                low_j = find_low(&reduced, j);
            } else {
                break;
            }
        }
        if let Some(low) = low_j {
            low_to_col[low] = Some(j);
        }
    }

    // Extract pairs
    let mut pairs = Vec::new();
    let mut paired_as_death = vec![false; n];

    for j in 0..n {
        if let Some(low) = find_low(&reduced, j) {
            pairs.push((low, Some(j)));
            paired_as_death[low] = true;
        }
    }

    // Unpaired columns are essential cycles (persist to infinity)
    for j in 0..n {
        if find_low(&reduced, j).is_none() && !paired_as_death[j] {
            pairs.push((j, None));
        }
    }

    ReductionResult { reduced, pairs }
}

fn find_low(matrix: &[Vec<i32>], j: usize) -> Option<usize> {
    matrix[j].iter().enumerate().rev()
        .find(|(_, v)| **v != 0)
        .map(|(i, _)| i)
}

/// Convert reduction result to barcodes using filtration values from the VR complex.
pub fn pairs_to_barcodes(
    result: &ReductionResult,
    complex: &VRComplex,
    max_dim: usize,
) -> Vec<Barcode> {
    let mut barcodes = Vec::new();

    for dim in 0..=max_dim {
        let mut bars = Vec::new();
        for &(birth_idx, death_idx) in &result.pairs {
            let birth_dim = complex.simplices[birth_idx].len().saturating_sub(1);
            if birth_dim != dim {
                continue;
            }
            let birth_val = complex.filtration_values[birth_idx];
            let death_val = death_idx.map(|d| complex.filtration_values[d]);
            match death_val {
                Some(dv) => {
                    if dv > birth_val + 1e-12 {
                        bars.push((birth_val, dv));
                    }
                }
                None => {
                    bars.push((birth_val, f64::INFINITY));
                }
            }
        }
        bars.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        barcodes.push(Barcode { dimension: dim, bars });
    }

    barcodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point_cloud::{ActionPoint, Metric, PointCloud};
    use crate::vietoris_rips::VRComplex;

    #[test]
    fn test_reduce_single_point() {
        let pc = PointCloud::new(
            vec![ActionPoint::new("a", 0.0, vec![0.0])],
            Metric::Euclidean,
        );
        let vr = VRComplex::build_filtration(&pc, 0);
        let bm = BoundaryMatrix::from_vr_complex(&vr);
        let result = reduce(&bm);
        let barcodes = pairs_to_barcodes(&result, &vr, 0);
        // Single point: H₀ has one bar [0, ∞)
        assert_eq!(barcodes.len(), 1);
        assert_eq!(barcodes[0].dimension, 0);
        assert_eq!(barcodes[0].bars.len(), 1);
        assert!((barcodes[0].bars[0].0 - 0.0).abs() < 1e-9);
        assert!(barcodes[0].bars[0].1.is_infinite());
    }

    #[test]
    fn test_reduce_two_points_merge() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![1.0]),
            ],
            Metric::Euclidean,
        );
        let vr = VRComplex::build_filtration(&pc, 1);
        let bm = BoundaryMatrix::from_vr_complex(&vr);
        let result = reduce(&bm);
        let barcodes = pairs_to_barcodes(&result, &vr, 1);
        // Two points: H₀ has one bar [0, ∞) and one bar [0, 1.0)
        let h0 = &barcodes[0];
        assert_eq!(h0.bars.len(), 2);
        let inf_bars: Vec<_> = h0.bars.iter().filter(|b| b.1.is_infinite()).collect();
        assert_eq!(inf_bars.len(), 1);
        let finite_bars: Vec<_> = h0.bars.iter().filter(|b| !b.1.is_infinite()).collect();
        assert_eq!(finite_bars.len(), 1);
        assert!((finite_bars[0].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_reduce_three_points_triangle() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![1.0]),
                ActionPoint::new("c", 2.0, vec![0.5]),
            ],
            Metric::Euclidean,
        );
        let vr = VRComplex::build_filtration(&pc, 2);
        let bm = BoundaryMatrix::from_vr_complex(&vr);
        let result = reduce(&bm);
        assert!(bm.verify_boundary_squared_zero());
        // Should have valid pairs
        assert!(!result.pairs.is_empty());
    }
}
