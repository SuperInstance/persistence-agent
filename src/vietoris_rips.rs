use serde::{Deserialize, Serialize};
use crate::point_cloud::PointCloud;

/// A Vietoris-Rips simplicial complex built from a point cloud.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRComplex {
    /// Each simplex is a sorted list of vertex indices.
    pub simplices: Vec<Vec<usize>>,
    /// Filtration value at which each simplex first appears.
    pub filtration_values: Vec<f64>,
}

impl VRComplex {
    /// Build the Vietoris-Rips complex up to max_dim simplices.
    ///
    /// A simplex σ is included when all pairwise distances among its vertices
    /// are ≤ ε, and the filtration value is the maximum such pairwise distance.
    pub fn build(cloud: &PointCloud, eps: f64, max_dim: usize) -> Self {
        let n = cloud.points.len();
        let dm = cloud.distance_matrix();
        let mut simplices = Vec::new();
        let mut filt_vals = Vec::new();

        // Vertices always present at filtration 0
        for i in 0..n {
            simplices.push(vec![i]);
            filt_vals.push(0.0);
        }

        // Iteratively build higher-dimensional simplices
        // Start with edges (dim 1)
        let mut current_simplices: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        for dim in 1..=max_dim {
            let mut next_simplices = Vec::new();
            let candidates = if dim == 1 {
                // Edges: all pairs
                let mut edges = Vec::new();
                for i in 0..n {
                    for j in (i + 1)..n {
                        edges.push(vec![i, j]);
                    }
                }
                edges
            } else {
                // Higher simplices: build from lower simplices
                Self::build_higher_simplices(&current_simplices, dim)
            };

            for sigma in candidates {
                // Check if max pairwise distance ≤ eps
                let max_d = Self::max_pairwise_distance(&sigma, &dm);
                if max_d <= eps + 1e-12 {
                    let mut sorted_sigma = sigma.clone();
                    sorted_sigma.sort();
                    // Avoid duplicates
                    if !simplices.contains(&sorted_sigma) {
                        simplices.push(sorted_sigma.clone());
                        filt_vals.push(max_d);
                        next_simplices.push(sorted_sigma);
                    }
                }
            }

            current_simplices = next_simplices;
            if current_simplices.is_empty() {
                break;
            }
        }

        // Sort by filtration value, then by simplex size, then lexicographically
        let mut combined: Vec<_> = simplices.into_iter().zip(filt_vals.into_iter()).collect();
        combined.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.len().cmp(&b.0.len()))
                .then_with(|| a.0.cmp(&b.0))
        });

        let (simplices, filtration_values) = combined.into_iter().unzip();
        Self { simplices, filtration_values }
    }

    /// Build the full filtration: insert simplices at all unique epsilon values.
    pub fn build_filtration(cloud: &PointCloud, max_dim: usize) -> Self {
        let n = cloud.points.len();
        let dm = cloud.distance_matrix();
        let mut simplices: Vec<Vec<usize>> = Vec::new();
        let mut filt_vals: Vec<f64> = Vec::new();

        // Vertices
        for i in 0..n {
            simplices.push(vec![i]);
            filt_vals.push(0.0);
        }

        // Edges
        for i in 0..n {
            for j in (i + 1)..n {
                simplices.push(vec![i, j]);
                filt_vals.push(dm[i][j]);
            }
        }

        // Higher simplices (dim >= 2)
        if max_dim >= 2 {
            let edges: Vec<Vec<usize>> = simplices.iter().filter(|s| s.len() == 2).cloned().collect();
            let _vertices: Vec<Vec<usize>> = simplices.iter().filter(|s| s.len() == 1).cloned().collect();
            let mut prev_dim_simplices: Vec<Vec<usize>> = edges;

            for dim in 2..=max_dim {
                let candidates = Self::build_higher_simplices(&prev_dim_simplices, dim);
                let mut next_simplices = Vec::new();

                for sigma in candidates {
                    let mut sorted = sigma;
                    sorted.sort();
                    if simplices.contains(&sorted) {
                        continue;
                    }
                    let max_d = Self::max_pairwise_distance(&sorted, &dm);
                    simplices.push(sorted.clone());
                    filt_vals.push(max_d);
                    next_simplices.push(sorted);
                }
                prev_dim_simplices = next_simplices;
                if prev_dim_simplices.is_empty() {
                    break;
                }
            }
        }

        // Sort
        let mut combined: Vec<_> = simplices.into_iter().zip(filt_vals.into_iter()).collect();
        combined.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.len().cmp(&b.0.len()))
                .then_with(|| a.0.cmp(&b.0))
        });

        let (simplices, filtration_values) = combined.into_iter().unzip();
        Self { simplices, filtration_values }
    }

    fn build_higher_simplices(prev_simplices: &[Vec<usize>], dim: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let n = prev_simplices.len();
        for i in 0..n {
            for j in (i + 1)..n {
                // Two (dim)-simplices can form a (dim+1)-simplex if they share (dim) vertices
                let a = &prev_simplices[i];
                let b = &prev_simplices[j];
                if a.len() != dim || b.len() != dim {
                    continue;
                }
                // Check that they share (dim - 1) vertices and differ in exactly 1
                let shared: Vec<usize> = a.iter().filter(|x| b.contains(x)).copied().collect();
                if shared.len() == dim - 1 {
                    let mut merged: Vec<usize> = a.clone();
                    for &x in b {
                        if !merged.contains(&x) {
                            merged.push(x);
                        }
                    }
                    merged.sort();
                    if merged.len() == dim + 1 && !result.contains(&merged) {
                        result.push(merged);
                    }
                }
            }
        }
        result
    }

    fn max_pairwise_distance(simplex: &[usize], dm: &[Vec<f64>]) -> f64 {
        let mut max_d: f64 = 0.0;
        for i in 0..simplex.len() {
            for j in (i + 1)..simplex.len() {
                max_d = max_d.max(dm[simplex[i]][simplex[j]]);
            }
        }
        max_d
    }

    /// Get simplices of a specific dimension.
    pub fn simplices_of_dim(&self, dim: usize) -> Vec<(Vec<usize>, f64)> {
        self.simplices
            .iter()
            .zip(self.filtration_values.iter())
            .filter(|(s, _)| s.len() == dim + 1)
            .map(|(s, f)| (s.clone(), *f))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point_cloud::{ActionPoint, Metric, PointCloud};

    #[test]
    fn test_vr_eps_zero_vertices_only() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![1.0]),
                ActionPoint::new("c", 2.0, vec![3.0]),
            ],
            Metric::Euclidean,
        );
        let vr = VRComplex::build(&pc, 0.0, 2);
        assert_eq!(vr.simplices.len(), 3);
        assert!(vr.simplices.iter().all(|s| s.len() == 1));
    }

    #[test]
    fn test_vr_large_eps_fully_connected() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![1.0]),
                ActionPoint::new("c", 2.0, vec![2.0]),
            ],
            Metric::Euclidean,
        );
        let vr = VRComplex::build(&pc, 100.0, 2);
        // 3 vertices + 3 edges + 1 triangle = 7
        assert_eq!(vr.simplices.len(), 7);
        assert!(vr.simplices.iter().any(|s| s.len() == 3));
    }

    #[test]
    fn test_vr_filtration_triangle() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![1.0]),
                ActionPoint::new("c", 2.0, vec![3.0]),
            ],
            Metric::Euclidean,
        );
        let vr = VRComplex::build_filtration(&pc, 2);
        // Should have 3 vertices + 3 edges + 1 triangle = 7
        assert_eq!(vr.simplices.len(), 7);
        let triangle_fv = vr.simplices.iter().zip(vr.filtration_values.iter())
            .find(|(s, _)| s.len() == 3)
            .map(|(_, f)| *f)
            .unwrap();
        // Max pairwise distance in triangle: d(0,2) = 3.0
        assert!((triangle_fv - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_vr_partial_eps() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![1.0]),
                ActionPoint::new("c", 2.0, vec![10.0]),
            ],
            Metric::Euclidean,
        );
        // eps = 5: should connect a-b but not c
        let vr = VRComplex::build(&pc, 5.0, 2);
        // 3 vertices + 1 edge = 4
        assert_eq!(vr.simplices.len(), 4);
        assert_eq!(vr.simplices.iter().filter(|s| s.len() == 2).count(), 1);
    }
}
