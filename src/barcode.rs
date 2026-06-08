use serde::{Deserialize, Deserializer, Serialize, Serializer};
use crate::boundary::BoundaryMatrix;
use crate::error::PersistenceError;
use crate::vietoris_rips::VietorisRipsComplex;

fn serialize_death<S: Serializer>(death: &f64, s: S) -> Result<S::Ok, S::Error> {
    if death.is_infinite() {
        s.serialize_str("inf")
    } else {
        s.serialize_f64(*death)
    }
}

fn deserialize_death<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    use serde::de;
    struct DeathVisitor;
    impl<'de> de::Visitor<'de> for DeathVisitor {
        type Value = f64;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("f64 or \"inf\"") }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<f64, E> { Ok(v) }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<f64, E> {
            if v == "inf" { Ok(f64::INFINITY) } else { Err(E::custom("expected \"inf\"")) }
        }
    }
    d.deserialize_any(DeathVisitor)
}

/// A single bar in a persistence barcode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistencePair {
    pub birth: f64,
    #[serde(serialize_with = "serialize_death", deserialize_with = "deserialize_death")]
    pub death: f64,
    pub dimension: usize,
}

impl PersistencePair {
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }
}

/// The full persistence barcode for all dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Barcode {
    pub pairs: Vec<PersistencePair>,
    /// Betti curve sampled at each unique filtration value:
    /// (epsilon, [β₀, β₁, β₂, …])
    pub betti_curve: Vec<(f64, Vec<usize>)>,
}

impl Barcode {
    /// Compute the barcode from a Vietoris-Rips complex.
    pub fn compute(vr: &VietorisRipsComplex) -> Result<Self, PersistenceError> {
        let mut bm = BoundaryMatrix::build(vr)?;
        let low_map = bm.reduce();

        let n = vr.n_simplices();
        let mut pairs = Vec::new();

        for (&row, &col) in &low_map {
            let birth_dim = vr.simplex_dimension(row);
            // birth simplex is `row`, death simplex is `col`
            let birth_eps = vr.filtration_values[row];
            let death_eps = vr.filtration_values[col];
            pairs.push(PersistencePair {
                birth: birth_eps,
                death: death_eps,
                dimension: birth_dim,
            });
        }

        // Remove dead code: unpaired columns logic was unused

        // Find unpaired birth simplices: those that never appear as `low(j)` for any j
        let mut is_low_of_some_col = vec![false; n];
        for &row in low_map.keys() {
            is_low_of_some_col[row] = true;
        }

        for (i, is_low) in is_low_of_some_col.iter().enumerate() {
            if vr.simplex_dimension(i) == 0 && !is_low {
                // Vertex that never got paired — persists to infinity
                pairs.push(PersistencePair {
                    birth: vr.filtration_values[i],
                    death: f64::INFINITY,
                    dimension: 0,
                });
            }
        }

        // Sort by (dimension, birth)
        pairs.sort_by(|a, b| {
            a.dimension
                .cmp(&b.dimension)
                .then_with(|| a.birth.partial_cmp(&b.birth).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Compute Betti curve
        let max_dim = vr.max_dimension;
        let mut thresholds = vr.filtration_values.to_vec();
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        let betti_curve = compute_betti_curve(&pairs, &thresholds, max_dim);

        Ok(Self { pairs, betti_curve })
    }

    /// Pairs filtered by dimension.
    pub fn pairs_of_dimension(&self, dim: usize) -> Vec<&PersistencePair> {
        self.pairs.iter().filter(|p| p.dimension == dim).collect()
    }

    /// Maximum persistence value among finite pairs.
    pub fn max_persistence(&self) -> f64 {
        self.pairs
            .iter()
            .filter(|p| p.death.is_finite())
            .map(|p| p.persistence())
            .fold(0.0_f64, f64::max)
    }

    /// Persistence entropy: Shannon entropy of normalized persistence values.
    pub fn persistence_entropy(&self) -> f64 {
        let persistences: Vec<f64> = self
            .pairs
            .iter()
            .filter(|p| p.death.is_finite() && p.persistence() > 1e-12)
            .map(|p| p.persistence())
            .collect();
        if persistences.is_empty() {
            return 0.0;
        }
        let total: f64 = persistences.iter().sum();
        if total < 1e-12 {
            return 0.0;
        }
        persistences
            .iter()
            .map(|&p| {
                let q = p / total;
                if q > 1e-12 {
                    -q * q.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Betti numbers at the given epsilon value.
    pub fn betti_numbers_at(&self, eps: f64) -> Vec<usize> {
        if self.betti_curve.is_empty() {
            return vec![];
        }
        // Find the last entry with epsilon <= eps
        let mut result = vec![0usize; self.betti_curve[0].1.len()];
        for &(e, ref bettis) in &self.betti_curve {
            if e > eps {
                break;
            }
            result = bettis.clone();
        }
        result
    }
}

fn compute_betti_curve(
    pairs: &[PersistencePair],
    thresholds: &[f64],
    max_dim: usize,
) -> Vec<(f64, Vec<usize>)> {
    let n_dims = max_dim + 1;
    let mut curve = Vec::with_capacity(thresholds.len());

    for &eps in thresholds {
        let mut bettis = vec![0usize; n_dims];
        for pair in pairs {
            if pair.dimension < n_dims {
                let alive = pair.birth <= eps + 1e-12
                    && (pair.death > eps - 1e-12 || !pair.death.is_finite());
                if alive {
                    bettis[pair.dimension] += 1;
                }
            }
        }
        curve.push((eps, bettis));
    }

    curve
}
