use serde::{Deserialize, Serialize};
use crate::barcode::BarcodeCollection;

/// Agent personality features extracted from persistence barcodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFeatures {
    /// Stability: proportion of long-lived H₀ bars (connected components that persist).
    /// High stability → consistent, predictable agent behavior.
    pub stability: f64,
    /// Adaptability: density of short-lived H₁ bars (transient loops = behavioral flexibility).
    /// High adaptability → agent explores many behavioral patterns.
    pub adaptability: f64,
    /// Depth: H₂ persistence (voids in behavior space = complex, multi-dimensional behavior).
    /// High depth → rich, layered behavioral repertoire.
    pub depth: f64,
    /// Combined personality vector [stability, adaptability, depth].
    pub personality_vector: Vec<f64>,
}

impl AgentFeatures {
    /// Extract agent features from a barcode collection.
    pub fn from_barcodes(barcodes: &BarcodeCollection) -> Self {
        let stability = Self::compute_stability(barcodes);
        let adaptability = Self::compute_adaptability(barcodes);
        let depth = Self::compute_depth(barcodes);
        let personality_vector = vec![stability, adaptability, depth];
        Self { stability, adaptability, depth, personality_vector }
    }

    /// Stability: ratio of infinite H₀ bars to total H₀ bars.
    /// An agent with many persistent components has stable behavior.
    fn compute_stability(barcodes: &BarcodeCollection) -> f64 {
        if let Some(h0) = barcodes.dimension(0) {
            if h0.bars.is_empty() {
                return 0.0;
            }
            let infinite = h0.bars.iter().filter(|(_, d)| d.is_infinite()).count();
            infinite as f64 / h0.bars.len() as f64
        } else {
            0.0
        }
    }

    /// Adaptability: (number of short-lived H₁ bars) / (total bars + 1).
    /// Short-lived loops indicate the agent cycles through behaviors quickly.
    fn compute_adaptability(barcodes: &BarcodeCollection) -> f64 {
        if let Some(h1) = barcodes.dimension(1) {
            let total = h1.bars.len();
            let short_lived = h1.bars.iter()
                .filter(|(b, d)| d.is_finite() && (d - b) < h1.mean_persistence())
                .count();
            short_lived as f64 / (total + 1) as f64
        } else {
            0.0
        }
    }

    /// Depth: total H₂ persistence (sum of void bar lengths).
    /// Voids in behavior space represent complex multi-dimensional patterns.
    fn compute_depth(barcodes: &BarcodeCollection) -> f64 {
        if let Some(h2) = barcodes.dimension(2) {
            h2.total_persistence()
        } else {
            0.0
        }
    }

    /// Classify the agent's behavioral archetype based on features.
    pub fn archetype(&self) -> &'static str {
        if self.stability > 0.7 && self.adaptability < 0.3 {
            "Steady"
        } else if self.adaptability > 0.5 && self.depth > 0.5 {
            "Explorer"
        } else if self.depth > 0.7 {
            "Deep"
        } else if self.stability > 0.5 && self.adaptability > 0.3 {
            "Balanced"
        } else {
            "Volatile"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::barcode::Barcode;

    #[test]
    fn test_features_stable_agent() {
        let barcodes = BarcodeCollection::new(vec![
            Barcode { dimension: 0, bars: vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)] },
            Barcode { dimension: 1, bars: vec![] },
        ]);
        let features = AgentFeatures::from_barcodes(&barcodes);
        assert!((features.stability - 1.0).abs() < 1e-9);
        assert_eq!(features.archetype(), "Steady");
    }

    #[test]
    fn test_features_adaptable_agent() {
        let barcodes = BarcodeCollection::new(vec![
            Barcode { dimension: 0, bars: vec![(0.0, f64::INFINITY)] },
            Barcode {
                dimension: 1,
                bars: vec![
                    (0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8),
                    (1.0, 2.0), (1.5, 2.5),
                ],
            },
            Barcode { dimension: 2, bars: vec![(0.5, 1.5), (1.0, 2.0)] },
        ]);
        let features = AgentFeatures::from_barcodes(&barcodes);
        assert!(features.adaptability > 0.0);
        assert!(features.depth > 0.0);
        assert_eq!(features.personality_vector.len(), 3);
    }

    #[test]
    fn test_features_empty_barcodes() {
        let barcodes = BarcodeCollection::new(vec![]);
        let features = AgentFeatures::from_barcodes(&barcodes);
        assert!((features.stability - 0.0).abs() < 1e-9);
        assert!((features.adaptability - 0.0).abs() < 1e-9);
        assert!((features.depth - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_archetype_balanced() {
        let features = AgentFeatures {
            stability: 0.6,
            adaptability: 0.4,
            depth: 0.2,
            personality_vector: vec![0.6, 0.4, 0.2],
        };
        assert_eq!(features.archetype(), "Balanced");
    }

    #[test]
    fn test_archetype_volatile() {
        let features = AgentFeatures {
            stability: 0.1,
            adaptability: 0.1,
            depth: 0.1,
            personality_vector: vec![0.1, 0.1, 0.1],
        };
        assert_eq!(features.archetype(), "Volatile");
    }

    #[test]
    fn test_serde_roundtrip_agent_features() {
        let features = AgentFeatures {
            stability: 0.75,
            adaptability: 0.5,
            depth: 0.3,
            personality_vector: vec![0.75, 0.5, 0.3],
        };
        let json = serde_json::to_string(&features).unwrap();
        let decoded: AgentFeatures = serde_json::from_str(&json).unwrap();
        assert!((decoded.stability - features.stability).abs() < 1e-9);
        assert!((decoded.adaptability - features.adaptability).abs() < 1e-9);
        assert_eq!(decoded.personality_vector, features.personality_vector);
    }
}
