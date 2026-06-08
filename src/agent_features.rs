use serde::{Deserialize, Serialize};
use crate::barcode::Barcode;
use crate::point_cloud::PointCloud;
use crate::vietoris_rips::VietorisRipsComplex;

/// Topological archetype of an agent's behavior pattern.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentArchetype {
    /// Single persistent cluster — stable, predictable behavior.
    Steady,
    /// Many short-lived loops — exploration-heavy, wandering.
    Explorer,
    /// Many disconnected components — volatile, switching between modes.
    Volatile,
    /// Long-lived higher-dimensional features — complex, structured behavior.
    Deep,
    /// Mix of features — no dominant topological signature.
    Balanced,
}

impl std::fmt::Display for AgentArchetype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentArchetype::Steady => write!(f, "Steady"),
            AgentArchetype::Explorer => write!(f, "Explorer"),
            AgentArchetype::Volatile => write!(f, "Volatile"),
            AgentArchetype::Deep => write!(f, "Deep"),
            AgentArchetype::Balanced => write!(f, "Balanced"),
        }
    }
}

/// A topological profile of an agent's behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub archetype: AgentArchetype,
    pub persistence_entropy: f64,
    pub max_persistence: f64,
    pub betti_numbers: Vec<usize>,
}

/// Profiler that maps topological features to agent behavior archetypes.
pub struct AgentProfiler {
    pub max_dimension: usize,
}

impl AgentProfiler {
    pub fn new(max_dimension: usize) -> Self {
        Self { max_dimension }
    }

    /// Profile an agent from its state-space trajectory (sequence of observation vectors).
    pub fn profile(&self, observations: Vec<Vec<f64>>) -> Result<AgentProfile, crate::error::PersistenceError> {
        let cloud = PointCloud::new(observations)?;
        let max_eps = cloud.max_distance();
        let vr = VietorisRipsComplex::build(&cloud, self.max_dimension, max_eps)?;
        let barcode = Barcode::compute(&vr)?;

        let entropy = barcode.persistence_entropy();
        let max_pers = barcode.max_persistence();
        let betti = barcode.betti_numbers_at(max_eps / 2.0);

        let archetype = self.classify(&barcode, &betti, cloud.n_points());

        Ok(AgentProfile {
            archetype,
            persistence_entropy: entropy,
            max_persistence: max_pers,
            betti_numbers: betti,
        })
    }

    fn classify(
        &self,
        barcode: &Barcode,
        _betti: &[usize],
        n_points: usize,
    ) -> AgentArchetype {
        let h0_pairs: Vec<_> = barcode.pairs_of_dimension(0);
        let h1_pairs: Vec<_> = barcode.pairs_of_dimension(1);

        let n_components = h0_pairs.len();
        // Long-lived H0 features (persistence > half of max)
        let max_pers = barcode.max_persistence();
        let long_h0 = h0_pairs
            .iter()
            .filter(|p| p.death.is_finite() && p.persistence() > max_pers * 0.5)
            .count();

        // Short-lived H1 features (persistence < 0.3 * max)
        let short_h1 = h1_pairs
            .iter()
            .filter(|p| p.death.is_finite() && p.persistence() < max_pers * 0.3)
            .count();

        // Long-lived H1+ features
        let higher_dim_count: usize = barcode
            .pairs
            .iter()
            .filter(|p| p.dimension >= 1 && p.death.is_finite() && p.persistence() > max_pers * 0.4)
            .count();

        // Many disconnected components relative to point count → Volatile
        let component_ratio = n_components as f64 / n_points.max(1) as f64;
        if component_ratio > 0.5 && long_h0 <= 1 {
            return AgentArchetype::Volatile;
        }

        // Single persistent cluster (1 long H0, few other features)
        if long_h0 == 1 && h1_pairs.len() <= 1 && higher_dim_count == 0 {
            return AgentArchetype::Steady;
        }

        // Many short-lived loops
        if short_h1 >= 3 || (h1_pairs.len() as f64 / n_points.max(1) as f64 > 0.3 && short_h1 >= 1) {
            return AgentArchetype::Explorer;
        }

        // Long-lived higher-dimensional features
        if higher_dim_count >= 1 {
            return AgentArchetype::Deep;
        }

        AgentArchetype::Balanced
    }
}
