use serde::{Deserialize, Serialize};
use crate::vietoris_rips::VRComplex;

/// A boundary matrix for a chain complex C₂ → C₁ → C₀.
///
/// Stored as a list of column vectors over Z₂ (hence `i32` values are 0 or 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryMatrix {
    /// Column-major: `matrix[j]` is column j of the boundary matrix.
    pub matrix: Vec<Vec<i32>>,
    /// Dimension of the simplicial complex used to build this.
    pub dim: usize,
    /// Number of columns (one per simplex).
    pub n_cols: usize,
    /// Number of rows (one per simplex in the boundary).
    pub n_rows: usize,
}

impl BoundaryMatrix {
    /// Build the full boundary matrix from a VR complex.
    ///
    /// Columns are indexed by simplex index (in the order given by VRComplex),
    /// and rows are indexed by simplex index as well. An entry `matrix[j][i] = 1`
    /// means simplex i is a face of simplex j.
    pub fn from_vr_complex(complex: &VRComplex) -> Self {
        let n = complex.simplices.len();
        let mut matrix = vec![vec![0i32; n]; n];

        for (j, sigma) in complex.simplices.iter().enumerate() {
            if sigma.len() <= 1 {
                // Vertices have empty boundary
                continue;
            }
            // Boundary of σ is all (dim-1)-faces: remove one vertex at a time
            for skip in 0..sigma.len() {
                let face: Vec<usize> = sigma.iter().enumerate()
                    .filter(|(k, _)| *k != skip)
                    .map(|(_, &v)| v)
                    .collect();
                // Find the index of this face in the complex
                if let Some(i) = complex.simplices.iter().position(|s| *s == face) {
                    matrix[j][i] = 1;
                }
            }
        }

        let dim = complex.simplices.iter().map(|s| s.len()).max().unwrap_or(1).saturating_sub(1);
        BoundaryMatrix { matrix, dim, n_cols: n, n_rows: n }
    }

    /// Verify ∂² = 0: the boundary of a boundary is zero.
    /// Returns true if ∂² = 0 holds over Z₂.
    pub fn verify_boundary_squared_zero(&self) -> bool {
        // Compute self * self over Z₂
        for i in 0..self.n_rows {
            for j in 0..self.n_cols {
                let mut sum = 0i32;
                for k in 0..self.n_cols {
                    sum += self.matrix[k][i] * self.matrix[j][k];
                }
                if sum % 2 != 0 {
                    return false;
                }
            }
        }
        true
    }

    /// Get the lowest nonzero row index in column j (returns None if column is zero).
    pub fn low(&self, j: usize) -> Option<usize> {
        self.matrix[j].iter().enumerate().rev()
            .find(|(_, v)| **v != 0)
            .map(|(i, _)| i)
    }

    /// Get the dimension of simplex at index i.
    pub fn simplex_dim(&self, complex: &VRComplex, i: usize) -> usize {
        complex.simplices[i].len().saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point_cloud::{ActionPoint, Metric, PointCloud};
    use crate::vietoris_rips::VRComplex;

    #[test]
    fn test_boundary_squared_zero_triangle() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![1.0]),
                ActionPoint::new("c", 2.0, vec![2.0]),
            ],
            Metric::Euclidean,
        );
        let vr = VRComplex::build_filtration(&pc, 2);
        let bm = BoundaryMatrix::from_vr_complex(&vr);
        assert!(bm.verify_boundary_squared_zero());
    }

    #[test]
    fn test_boundary_squared_zero_complex() {
        // Four points in a tetrahedron-like arrangement
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0, 0.0]),
                ActionPoint::new("b", 1.0, vec![1.0, 0.0]),
                ActionPoint::new("c", 2.0, vec![0.0, 1.0]),
                ActionPoint::new("d", 3.0, vec![1.0, 1.0]),
            ],
            Metric::Euclidean,
        );
        let vr = VRComplex::build_filtration(&pc, 2);
        let bm = BoundaryMatrix::from_vr_complex(&vr);
        assert!(bm.verify_boundary_squared_zero());
    }

    #[test]
    fn test_boundary_vertex_is_zero_column() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![1.0]),
            ],
            Metric::Euclidean,
        );
        let vr = VRComplex::build_filtration(&pc, 1);
        let bm = BoundaryMatrix::from_vr_complex(&vr);
        // First two columns (vertices) should be zero
        assert!(bm.matrix[0].iter().all(|&v| v == 0));
        assert!(bm.matrix[1].iter().all(|&v| v == 0));
    }

    #[test]
    fn test_boundary_edge_has_two_vertices() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![1.0]),
            ],
            Metric::Euclidean,
        );
        let vr = VRComplex::build_filtration(&pc, 1);
        let bm = BoundaryMatrix::from_vr_complex(&vr);
        // Edge column should have exactly two 1s
        let edge_col = &bm.matrix[2]; // third simplex is the edge
        let ones = edge_col.iter().filter(|&&v| v != 0).count();
        assert_eq!(ones, 2);
    }
}
