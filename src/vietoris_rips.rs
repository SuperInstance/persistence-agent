use crate::error::PersistenceError;
use crate::point_cloud::PointCloud;

/// A Vietoris-Rips complex built from a point cloud.
/// Simplices are added as epsilon grows — a k-simplex [v0,…,vk] appears at the
/// maximum pairwise distance among its vertices.
#[derive(Debug, Clone)]
pub struct VietorisRipsComplex {
    pub simplices: Vec<Vec<usize>>,
    pub filtration_values: Vec<f64>,
    pub max_dimension: usize,
}

impl VietorisRipsComplex {
    /// Build the Vietoris-Rips complex up to `max_dimension`.
    /// For large point clouds you can pass `max_eps` to prune simplices whose
    /// filtration value exceeds it; pass `f64::INFINITY` to include everything.
    pub fn build(
        cloud: &PointCloud,
        max_dimension: usize,
        max_eps: f64,
    ) -> Result<Self, PersistenceError> {
        let n = cloud.n_points();
        if n == 0 {
            return Err(PersistenceError::EmptyCloud);
        }

        let mut simplices: Vec<Vec<usize>> = Vec::new();
        let mut filtration_values: Vec<f64> = Vec::new();

        // Dimension 0: each vertex appears at epsilon = 0
        for i in 0..n {
            simplices.push(vec![i]);
            filtration_values.push(0.0);
        }

        // Iteratively build higher-dimensional simplices
        let mut current_simplices: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        for _dim in 0..max_dimension {
            let mut candidates: Vec<(f64, Vec<usize>)> = Vec::new();
            for simplex in &current_simplices {
                // Try to extend by adding any vertex with index > max(simplex)
                let max_v = *simplex.iter().max().unwrap();
                for v in (max_v + 1)..n {
                    let mut new_simplex = simplex.clone();
                    new_simplex.push(v);
                    new_simplex.sort();

                    // Filtration value = max pairwise distance
                    let filt = max_pairwise(cloud, &new_simplex);
                    if filt <= max_eps {
                        candidates.push((filt, new_simplex));
                    }
                }
            }

            // Sort by filtration value, break ties by lexicographic order
            candidates.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });

            current_simplices = candidates.iter().map(|(_, s)| s.clone()).collect();
            for (filt, s) in &candidates {
                simplices.push(s.clone());
                filtration_values.push(*filt);
            }

            if current_simplices.is_empty() {
                break;
            }
        }

        Ok(Self {
            simplices,
            filtration_values,
            max_dimension,
        })
    }

    pub fn n_simplices(&self) -> usize {
        self.simplices.len()
    }

    /// Return the dimension of a simplex at given index.
    pub fn simplex_dimension(&self, idx: usize) -> usize {
        self.simplices[idx].len().saturating_sub(1)
    }

    /// Indices of all simplices of a given dimension.
    pub fn simplices_of_dimension(&self, dim: usize) -> Vec<usize> {
        self.simplices
            .iter()
            .enumerate()
            .filter(|(_, s)| s.len().saturating_sub(1) == dim)
            .map(|(i, _)| i)
            .collect()
    }

    /// All (dimension, simplex-index) pairs sorted by filtration value then dimension.
    pub fn sorted_filtration(&self) -> Vec<(usize, usize)> {
        let mut indexed: Vec<(usize, usize, f64)> = self
            .simplices
            .iter()
            .enumerate()
            .map(|(i, s)| (s.len().saturating_sub(1), i, self.filtration_values[i]))
            .collect();
        indexed.sort_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        indexed.into_iter().map(|(d, i, _)| (d, i)).collect()
    }
}

fn max_pairwise(cloud: &PointCloud, simplex: &[usize]) -> f64 {
    let mut mx = 0.0_f64;
    for i in 0..simplex.len() {
        for j in (i + 1)..simplex.len() {
            mx = mx.max(cloud.distance(simplex[i], simplex[j]));
        }
    }
    mx
}
