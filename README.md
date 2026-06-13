# persistence-agent

**Persistent homology for agent behavior.**

Agent actions form a point cloud in behavior space. Persistent homology detects which behavioral patterns are signal vs noise. Long-lived topological features = real agent personality. Short-lived = noise.

---

## Why?

When you observe an AI agent over time, its actions trace out a shape in high-dimensional behavior space. Some patterns persist — those are personality. Others flicker and vanish — that's noise.

This library applies **topological data analysis (TDA)** to that problem:

| Homology | Topological Feature | Behavioral Meaning |
|----------|--------------------|--------------------|
| H₀ | Connected components | **Stability** — how many distinct behavioral clusters persist |
| H₁ | Loops | **Adaptability** — cycles of behavioral exploration |
| H₂ | Voids | **Depth** — complex, multi-dimensional behavioral patterns |

A persistence barcode tells you which features are real (they survive across many scales) and which are artifacts (they appear and die quickly).

## Quick Start

```rust
use persistence_agent::{ActionPoint, Metric, analyze};

// Record agent actions as feature vectors
let actions = vec![
    ActionPoint::new("agent-1", 0.0, vec![0.8, 0.2, 0.1]),
    ActionPoint::new("agent-1", 1.0, vec![0.75, 0.25, 0.15]),
    ActionPoint::new("agent-1", 2.0, vec![0.1, 0.9, 0.8]),
    ActionPoint::new("agent-1", 3.0, vec![0.85, 0.15, 0.05]),
    ActionPoint::new("agent-1", 4.0, vec![0.2, 0.85, 0.75]),
];

// Run the full pipeline: point cloud → VR complex → boundary → reduce → barcode → features
let (barcodes, features) = analyze(actions, Metric::Cosine, 2);

// What kind of agent is this?
println!("Stability:     {:.3}", features.stability);    // long H₀ bars
println!("Adaptability:  {:.3}", features.adaptability);  // short H₁ bars
println!("Depth:         {:.3}", features.depth);         // H₂ persistence
println!("Archetype:     {}", features.archetype());       // Steady, Explorer, Deep, Balanced, Volatile

// Inspect the barcode
for bc in &barcodes.barcodes {
    let dim_name = match bc.dimension {
        0 => "H₀ (components)",
        1 => "H₁ (loops)",
        2 => "H₂ (voids)",
        _ => "Hₙ",
    };
    println!("{}: {} bars", dim_name, bc.num_bars());
    for (birth, death) in &bc.bars {
        let death_str = if death.is_infinite() { "∞".to_string() } else { format!("{:.3}", death) };
        println!("  [{:.3}, {})", birth, death_str);
    }
}
```

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│ ActionPoint │────▶│   PointCloud     │────▶│  VRComplex     │
│ (features)  │     │  (distance mat)  │     │  (simplices)   │
└─────────────┘     └──────────────────┘     └────────────────┘
                                                      │
                                                      ▼
┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│ AgentFeatures│◀────│    Barcode       │◀────│BoundaryMatrix  │
│ (personality)│     │  (birth, death)  │     │  + Reduction   │
└──────────────┘     └──────────────────┘     └────────────────┘
```

## Modules

### `point_cloud` — Actions as Points

Agent actions become vectors in behavior space. Three distance metrics:

- **Euclidean** — straight-line distance between action vectors
- **Cosine** — angle between vectors (ignores magnitude, focuses on direction)
- **Manhattan** — L1 distance (robust to outliers)

```rust
use persistence_agent::{ActionPoint, PointCloud, Metric};

let cloud = PointCloud::new(
    vec![
        ActionPoint::new("agent", 0.0, vec![1.0, 0.0, 0.0]),
        ActionPoint::new("agent", 1.0, vec![0.0, 1.0, 0.0]),
    ],
    Metric::Cosine,
);

let distance = cloud.distance(0, 1);  // 1.0 (orthogonal)
let neighbors = cloud.knn(0, 5);      // k-nearest neighbors
let matrix = cloud.distance_matrix(); // full pairwise distances
```

### `vietoris_rips` — Building the Complex

The Vietoris-Rips complex connects points within distance ε. As ε grows from 0 to ∞, simplices appear — this is the **filtration**.

```rust
use persistence_agent::{PointCloud, ActionPoint, Metric, VRComplex};

let cloud = PointCloud::new(points, Metric::Euclidean);
let vr = VRComplex::build_filtration(&cloud, 2); // up to 2-simplices (triangles)
let edges = vr.simplices_of_dim(1);
let triangles = vr.simplices_of_dim(2);
```

### `boundary` — Chain Complex

The boundary operator maps k-simplices to (k-1)-simplices. The fundamental property: **∂² = 0** — the boundary of a boundary is always empty.

```rust
use persistence_agent::{VRComplex, BoundaryMatrix};

let bm = BoundaryMatrix::from_vr_complex(&vr);
assert!(bm.verify_boundary_squared_zero()); // always true for valid complexes
```

### `reduction` — Standard Algorithm

Column reduction on the boundary matrix over Z₂ finds birth-death pairs. Each pair tells you when a topological feature appeared and when it was destroyed.

### `barcode` — Persistence Barcodes

A barcode is a set of intervals `[birth, death)` per homology dimension:

```rust
use persistence_agent::Barcode;

let h0 = Barcode { dimension: 0, bars: vec![(0.0, f64::INFINITY), (0.0, 2.5)] };
println!("Betti number at ε=1.0: {}", h0.betti_at(1.0)); // 2 components
println!("Betti number at ε=3.0: {}", h0.betti_at(3.0)); // 1 component (merged)

let curve = h0.betti_curve(&[0.0, 1.0, 2.0, 3.0]); // Betti numbers across scales
```

### `agent_features` — Personality Extraction

From barcodes to behavioral traits:

| Feature | Source | Meaning |
|---------|--------|---------|
| `stability` | H₀ infinite bars / total | Fraction of persistent behavioral clusters |
| `adaptability` | Short H₁ bars / total | Behavioral exploration rate |
| `depth` | H₂ total persistence | Complexity of behavioral repertoire |

Archetypes: **Steady** (stable, low adaptability), **Explorer** (adaptable + deep), **Deep** (high H₂), **Balanced** (moderate everything), **Volatile** (low on all metrics).

## The Math (Condensed)

1. **Point cloud** → pairwise distances using chosen metric
2. **Vietoris-Rips complex** → for each ε, connect all points within distance ε. A set of k+1 points forms a k-simplex if all pairwise distances ≤ ε
3. **Boundary matrix** → encode ∂: Cₖ → Cₖ₋₁ over Z₂. Each column lists the faces of a simplex
4. **Column reduction** → Gaussian elimination over Z₂. The "low" entries of reduced columns give birth-death pairs
5. **Barcode** → each pair (σ_birth, σ_death) becomes an interval [f(σ_birth), f(σ_death)) where f is the filtration value
6. **Betti numbers** → βₖ(ε) = number of k-dimensional bars alive at scale ε

## Serde Support

All public types derive `Serialize` and `Deserialize`:

```rust
use persistence_agent::ActionPoint;

let action = ActionPoint::new("agent-7", 1234567890.0, vec![0.5, 0.3, 0.2]);
let json = serde_json::to_string(&action)?;
let decoded: ActionPoint = serde_json::from_str(&json)?;
```

## Performance Notes

- The VR complex grows combinatorially with the number of points. For N points, there are up to 2ᴺ simplices
- Practical for N ≤ ~20 with max_dim ≤ 3. For larger point clouds, consider sparse filtrations or witness complexes (not yet implemented)
- Column reduction is O(N³) in the worst case — fine for small complexes, prohibitive for large ones

## License

MIT
