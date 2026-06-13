use serde::{Deserialize, Serialize};

/// A persistence barcode for a single homology dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Barcode {
    /// Homology dimension (0 = connected components, 1 = loops, 2 = voids).
    pub dimension: usize,
    /// Each bar is a (birth, death) pair. death = ∞ means the feature persists.
    pub bars: Vec<(f64, f64)>,
}

impl Barcode {
    /// Number of bars.
    pub fn num_bars(&self) -> usize {
        self.bars.len()
    }

    /// Number of bars alive at a given epsilon value.
    pub fn betti_at(&self, eps: f64) -> usize {
        self.bars.iter()
            .filter(|(birth, death)| *birth <= eps + 1e-12 && (death.is_infinite() || *death > eps))
            .count()
    }

    /// Compute the Betti curve: Betti number at each epsilon threshold.
    pub fn betti_curve(&self, eps_values: &[f64]) -> Vec<usize> {
        eps_values.iter().map(|&eps| self.betti_at(eps)).collect()
    }

    /// Mean persistence (average bar length) for finite bars.
    pub fn mean_persistence(&self) -> f64 {
        let finite_bars: Vec<_> = self.bars.iter().filter(|(_, d)| d.is_finite()).collect();
        if finite_bars.is_empty() {
            return 0.0;
        }
        finite_bars.iter().map(|(b, d)| d - b).sum::<f64>() / finite_bars.len() as f64
    }

    /// Maximum bar length among finite bars.
    pub fn max_persistence(&self) -> f64 {
        self.bars.iter()
            .filter(|(_, d)| d.is_finite())
            .map(|(b, d)| d - b)
            .fold(0.0f64, f64::max)
    }

    /// Total persistence: sum of all finite bar lengths.
    pub fn total_persistence(&self) -> f64 {
        self.bars.iter()
            .filter(|(_, d)| d.is_finite())
            .map(|(b, d)| d - b)
            .sum()
    }
}

/// A collection of barcodes across all homology dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarcodeCollection {
    pub barcodes: Vec<Barcode>,
}

impl BarcodeCollection {
    pub fn new(barcodes: Vec<Barcode>) -> Self {
        Self { barcodes }
    }

    /// Get the barcode for a specific dimension.
    pub fn dimension(&self, dim: usize) -> Option<&Barcode> {
        self.barcodes.iter().find(|b| b.dimension == dim)
    }

    /// Compute Betti numbers at a given epsilon.
    pub fn betti_numbers(&self, eps: f64) -> Vec<usize> {
        self.barcodes.iter().map(|b| b.betti_at(eps)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_point_h0_infinite() {
        let barcode = Barcode {
            dimension: 0,
            bars: vec![(0.0, f64::INFINITY)],
        };
        assert_eq!(barcode.num_bars(), 1);
        assert_eq!(barcode.betti_at(0.0), 1);
        assert_eq!(barcode.betti_at(100.0), 1);
    }

    #[test]
    fn test_two_points_merge() {
        // Two points at distance 2.0: H₀ bars [0,∞) and [0,2)
        let barcode = Barcode {
            dimension: 0,
            bars: vec![(0.0, f64::INFINITY), (0.0, 2.0)],
        };
        assert_eq!(barcode.betti_at(0.0), 2); // two components at ε=0
        assert_eq!(barcode.betti_at(1.0), 2); // still two
        assert_eq!(barcode.betti_at(2.0), 1); // merged
        assert_eq!(barcode.betti_at(10.0), 1);
    }

    #[test]
    fn test_betti_curve() {
        let barcode = Barcode {
            dimension: 0,
            bars: vec![(0.0, f64::INFINITY), (0.0, 3.0)],
        };
        let curve = barcode.betti_curve(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(curve, vec![2, 2, 2, 1, 1, 1]);
    }

    #[test]
    fn test_mean_persistence() {
        let barcode = Barcode {
            dimension: 0,
            bars: vec![(0.0, 2.0), (0.0, 4.0)],
        };
        let mean = barcode.mean_persistence();
        assert!((mean - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_persistence() {
        let barcode = Barcode {
            dimension: 1,
            bars: vec![(1.0, 3.0), (2.0, 5.0), (0.5, 0.7)],
        };
        assert!((barcode.max_persistence() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_total_persistence() {
        let barcode = Barcode {
            dimension: 0,
            bars: vec![(0.0, 1.0), (0.0, 2.0), (0.0, f64::INFINITY)],
        };
        assert!((barcode.total_persistence() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_barcode_collection_betti_numbers() {
        let collection = BarcodeCollection::new(vec![
            Barcode { dimension: 0, bars: vec![(0.0, f64::INFINITY), (0.0, 2.0)] },
            Barcode { dimension: 1, bars: vec![(1.5, 3.0)] },
        ]);
        let betti = collection.betti_numbers(1.0);
        assert_eq!(betti, vec![2, 0]); // 2 components, 0 loops
        let betti = collection.betti_numbers(2.0);
        assert_eq!(betti, vec![1, 1]); // 1 component, 1 loop
    }
}
