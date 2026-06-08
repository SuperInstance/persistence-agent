# persistence-agent

[![crates.io](https://img.shields.io/crates/v/persistence-agent.svg)](https://crates.io/crates/persistence-agent)
[![docs.rs](https://docs.rs/persistence-agent/badge.svg)](https://docs.rs/persistence-agent)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The Problem

An agent's behavior over time produces a sequence of states — a point cloud in some high-dimensional state space. How do you characterize the *shape* of that behavior? Is the agent steady (stays in one region), exploratory (wanders widely), cyclic (returns to previous states), or chaotic (fills a high-dimensional region)?

Time-series statistics (mean, variance, autocorrelation) capture quantitative aspects but miss the *topology*: whether the behavior has loops, clusters, voids, or more exotic shapes. Two agents can have identical mean/variance but fundamentally different behavioral topology.

## The Idea: Persistent Homology

**Persistent homology** computes the topological features of a point cloud across all scales simultaneously. You grow balls around each point. When balls overlap, you connect them. As the balls grow:
- First, isolated points merge into one connected component (β₀ drops)
- Then, loops form and fill in (β₁ rises and falls)
- Higher-dimensional features appear and disappear (β₂)

The result is a **barcode**: for each topological feature, a bar [birth, death] showing the range of scales at which it exists. Long bars = real structure. Short bars = noise.

### What the topology means for agents

| Topological feature | Behavioral meaning |
|---|---|
| One persistent component (β₀=1) | Agent stays in one behavioral mode |
| Two components (β₀=2) | Agent switches between two distinct modes |
| Persistent loop (β₁=1) | Cyclic behavior — agent returns to previous states |
| Many short-lived loops | Exploration — agent visits many configurations temporarily |
| High-dimensional voids (β₂>0) | Complex behavior with internal structure |

## How It Works

### Build a point cloud from agent states

```rust
use persistence_agent::PointCloud;

// Each Vec<f64> is one agent state observation
let cloud = PointCloud::from_vectors(&[
    vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.87],  // triangle
    vec![0.5, -0.87], vec![1.5, 0.0], vec![1.0, -0.87], // second triangle
]);
```

### Build Vietoris-Rips complex

```rust
use persistence_agent::VietorisRipsComplex;

let complex = VietorisRipsComplex::build(&cloud, /* max_dim */ 2);
// As epsilon grows, simplices are added in filtration order
println!("{} simplices, max filtration value: {:.3}",
    complex.simplices.len(), complex.max_filtration());
```

### Compute the barcode

```rust
use persistence_agent::Barcode;

let barcode = Barcode::from_complex(&complex);
for pair in &barcode.pairs {
    println!("Dim {}: [{:.3}, {:.3}] (persistence: {:.3})",
        pair.dimension, pair.birth, pair.death, pair.death - pair.birth);
}
```

### Betti curves

```rust
for (epsilon, betti) in &barcode.betti_curve {
    println!("ε={:.2}: β₀={} β₁={} β₂={}",
        epsilon, betti[0], betti[1], betti[2]);
}
```

### Classify agent archetype

```rust
use persistence_agent::agent_features::AgentProfiler;

let profile = AgentProfiler::profile(&barcode);
println!("Archetype: {:?}", profile.archetype);
// Steady    = single persistent cluster
// Explorer  = many short-lived loops
// Volatile  = many disconnected components
// Deep      = long-lived higher-dimensional features
// Balanced  = mix of features

println!("Persistence entropy: {:.3}", profile.persistence_entropy);
println!("Max persistence: {:.3}", profile.max_persistence);
```

**Persistence entropy** H = -Σ pᵢ log(pᵢ) where pᵢ is the persistence of feature i divided by total persistence. Low H = one feature dominates (steady agent). High H = features are equally important (complex behavior).

## Verified Test Cases

| Point cloud shape | Expected topology | What it means for an agent |
|---|---|---|
| Single point | β₀=1 | Never moves |
| Line of points | β₀=1, no loops | Linear trajectory |
| Circle | β₀=1, β₁=1 | Perfectly cyclic behavior |
| Figure-eight | β₀=1, β₁=2 | Two intertwined cycles |
| Two clusters | β₀=2 | Bimodal: switches between two states |
| Random points | β₀=1, many short bars | Noisy/exploratory |

## Module Map

| Module | What it does |
|---|---|
| `point_cloud` | `PointCloud` — distance matrix, k-NN graph |
| `vietoris_rips` | `VietorisRipsComplex` — filtration builder |
| `boundary` | `BoundaryMatrix` — mod 2 column reduction |
| `barcode` | `Barcode`, `PersistencePair` — persistence diagram, Betti curves |
| `agent_features` | `AgentProfiler`, `AgentArchetype` — topological behavior classification |
| `error` | `PersistenceError` |

## Performance

For n points, the Vietoris-Rips complex has O(nᵈ) simplices where d is the maximum dimension. For d=2 (β₀ and β₁), this is O(n³). Practical limits:
- n < 200: runs in seconds
- n < 1000: runs in minutes
- n > 1000: use [witness-topology](https://crates.io/crates/witness-topology) for landmark-based approximation

## Links

- [Documentation](https://docs.rs/persistence-agent)
- [Repository](https://github.com/SuperInstance/persistence-agent)
- [crates.io](https://crates.io/crates/persistence-agent)
- Edelsbrunner & Harer (2010) — *Computational Topology: An Introduction*

## License

MIT
