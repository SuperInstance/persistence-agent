use crate::error::PersistenceError;

/// A set of points in Euclidean space with a precomputed distance matrix.
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<Vec<f64>>,
    pub distance_matrix: Vec<Vec<f64>>,
}

impl PointCloud {
    /// Build a point cloud, computing the pairwise Euclidean distance matrix.
    pub fn new(points: Vec<Vec<f64>>) -> Result<Self, PersistenceError> {
        if points.is_empty() {
            return Err(PersistenceError::EmptyCloud);
        }
        let dim = points[0].len();
        for p in points.iter() {
            if p.len() != dim {
                return Err(PersistenceError::DimensionMismatch {
                    expected: dim,
                    actual: p.len(),
                });
            }
        }
        let n = points.len();
        let mut dm = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = euclidean(&points[i], &points[j]);
                dm[i][j] = d;
                dm[j][i] = d;
            }
        }
        Ok(Self {
            points,
            distance_matrix: dm,
        })
    }

    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    pub fn dimension(&self) -> usize {
        self.points[0].len()
    }

    /// Distance between points at indices `i` and `j`.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        self.distance_matrix[i][j]
    }

    /// k-nearest-neighbor graph: returns for each point the indices of its k nearest
    /// neighbors (excluding itself), sorted by distance ascending.
    pub fn knn(&self, k: usize) -> Result<Vec<Vec<usize>>, PersistenceError> {
        let n = self.n_points();
        if k == 0 || k >= n {
            return Err(PersistenceError::InvalidK { k, n });
        }
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut pairs: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (self.distance_matrix[i][j], j))
                .collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            result.push(pairs[..k].iter().map(|(_, j)| *j).collect());
        }
        Ok(result)
    }

    /// Maximum pairwise distance in the cloud.
    pub fn max_distance(&self) -> f64 {
        let n = self.n_points();
        let mut mx: f64 = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                mx = mx.max(self.distance_matrix[i][j]);
            }
        }
        mx
    }

    /// Return the sorted list of unique pairwise distances (filtration thresholds).
    pub fn unique_distances(&self) -> Vec<f64> {
        let n = self.n_points();
        let mut dists = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                dists.push(self.distance_matrix[i][j]);
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        dists
    }
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}
