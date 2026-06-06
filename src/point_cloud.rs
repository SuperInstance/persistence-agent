use serde::{Deserialize, Serialize};

// ── Core types ──────────────────────────────────────────────────────────────

/// A single agent action represented as a point in behavior space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPoint {
    pub agent_id: String,
    pub timestamp: f64,
    pub features: Vec<f64>,
}

impl ActionPoint {
    pub fn new(agent_id: impl Into<String>, timestamp: f64, features: Vec<f64>) -> Self {
        Self { agent_id: agent_id.into(), timestamp, features }
    }

    /// L2 norm of the feature vector.
    pub fn norm(&self) -> f64 {
        self.features.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

/// Distance metric for comparing action points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    Euclidean,
    Cosine,
    Manhattan,
}

/// A collection of action points forming a point cloud in behavior space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloud {
    pub points: Vec<ActionPoint>,
    pub metric: Metric,
}

impl PointCloud {
    pub fn new(points: Vec<ActionPoint>, metric: Metric) -> Self {
        Self { points, metric }
    }

    /// Compute the distance between two points using the configured metric.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        let a = &self.points[i].features;
        let b = &self.points[j].features;
        match self.metric {
            Metric::Euclidean => {
                a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
            }
            Metric::Cosine => {
                let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    return 1.0; // max cosine distance for zero vectors
                }
                1.0 - dot / (norm_a * norm_b)
            }
            Metric::Manhattan => {
                a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
            }
        }
    }

    /// Compute the full pairwise distance matrix.
    pub fn distance_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.points.len();
        let mut dm = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.distance(i, j);
                dm[i][j] = d;
                dm[j][i] = d;
            }
        }
        dm
    }

    /// Find the k nearest neighbors of point `i` (excluding the point itself).
    pub fn knn(&self, i: usize, k: usize) -> Vec<(usize, f64)> {
        let n = self.points.len();
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, self.distance(i, j)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }

    /// Get all unique pairwise distances, sorted ascending.
    pub fn sorted_distances(&self) -> Vec<f64> {
        let n = self.points.len();
        let mut dists = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                dists.push(self.distance(i, j));
            }
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        dists
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0, 0.0]),
                ActionPoint::new("b", 1.0, vec![3.0, 4.0]),
            ],
            Metric::Euclidean,
        );
        let d = pc.distance(0, 1);
        assert!((d - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_distance_identical() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![1.0, 2.0, 3.0]),
                ActionPoint::new("b", 1.0, vec![2.0, 4.0, 6.0]),
            ],
            Metric::Cosine,
        );
        let d = pc.distance(0, 1);
        assert!(d.abs() < 1e-9); // same direction → distance 0
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![1.0, 0.0]),
                ActionPoint::new("b", 1.0, vec![0.0, 1.0]),
            ],
            Metric::Cosine,
        );
        let d = pc.distance(0, 1);
        assert!((d - 1.0).abs() < 1e-9); // orthogonal → distance 1
    }

    #[test]
    fn test_manhattan_distance() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![1.0, 2.0]),
                ActionPoint::new("b", 1.0, vec![4.0, 6.0]),
            ],
            Metric::Manhattan,
        );
        let d = pc.distance(0, 1);
        assert!((d - 7.0).abs() < 1e-9); // |3| + |4| = 7
    }

    #[test]
    fn test_distance_matrix_symmetry() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![1.0]),
                ActionPoint::new("c", 2.0, vec![3.0]),
            ],
            Metric::Euclidean,
        );
        let dm = pc.distance_matrix();
        for i in 0..3 {
            for j in 0..3 {
                assert!((dm[i][j] - dm[j][i]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_knn_basic() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![1.0]),
                ActionPoint::new("c", 2.0, vec![5.0]),
                ActionPoint::new("d", 3.0, vec![10.0]),
            ],
            Metric::Euclidean,
        );
        let nn = pc.knn(0, 2);
        assert_eq!(nn.len(), 2);
        assert_eq!(nn[0].0, 1); // closest to 0 is 1
        assert!((nn[0].1 - 1.0).abs() < 1e-9);
        assert_eq!(nn[1].0, 2); // second closest is 2
        assert!((nn[1].1 - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_sorted_distances() {
        let pc = PointCloud::new(
            vec![
                ActionPoint::new("a", 0.0, vec![0.0]),
                ActionPoint::new("b", 1.0, vec![2.0]),
                ActionPoint::new("c", 2.0, vec![5.0]),
            ],
            Metric::Euclidean,
        );
        let d = pc.sorted_distances();
        // Distances: (0,1)=2, (1,2)=3, (0,2)=5 → sorted: [2, 3, 5]
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-9);
        assert!((d[1] - 3.0).abs() < 1e-9);
        assert!((d[2] - 5.0).abs() < 1e-9);
    }
}
