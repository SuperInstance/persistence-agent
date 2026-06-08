//! # persistence-agent
//!
//! Persistent homology for agent behavior profiling — topological fingerprints.
//!
//! This crate computes persistent homology from point clouds representing agent
//! state trajectories and maps topological invariants to behavior archetypes.
//!
//! ## Quick Start
//!
//! ```
//! use persistence_agent::agent_features::AgentProfiler;
//!
//! // Agent observations over time (e.g. 2D state vectors)
//! let obs = vec![
//!     vec![0.0, 0.0],
//!     vec![0.1, 0.0],
//!     vec![0.0, 0.1],
//!     vec![1.0, 1.0],
//!     vec![1.1, 1.0],
//! ];
//!
//! let profiler = AgentProfiler::new(1);
//! let profile = profiler.profile(obs).unwrap();
//! println!("Archetype: {}", profile.archetype);
//! println!("Entropy:   {:.3}", profile.persistence_entropy);
//! ```
//!
//! ## Pipeline
//!
//! 1. **PointCloud** — embed observations, compute distances
//! 2. **VietorisRipsComplex** — build filtration from pairwise distances
//! 3. **BoundaryMatrix** — mod-2 boundary matrix + column reduction
//! 4. **Barcode** — extract persistence pairs and Betti curves
//! 5. **AgentProfiler** — classify behavior into archetypes
//!
//! ## Archetypes
//!
//! | Archetype | Topological Signature |
//! |-----------|----------------------|
//! | Steady    | Single persistent cluster (β₀ = 1) |
//! | Explorer  | Many short-lived loops |
//! | Volatile  | Many disconnected components |
//! | Deep      | Long-lived higher-dimensional features |
//! | Balanced  | Mixed features, no dominant signature |

pub mod agent_features;
pub mod barcode;
pub mod boundary;
pub mod error;
pub mod point_cloud;
pub mod vietoris_rips;

#[cfg(test)]
mod tests;

pub use agent_features::{AgentArchetype, AgentProfile, AgentProfiler};
pub use barcode::{Barcode, PersistencePair};
pub use boundary::BoundaryMatrix;
pub use error::PersistenceError;
pub use point_cloud::PointCloud;
pub use vietoris_rips::VietorisRipsComplex;
