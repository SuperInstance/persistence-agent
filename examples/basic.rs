//! Basic example: Persistent homology analysis of agent behavior.
//!
//! Run with: cargo run --example basic

use persistence_agent::{ActionPoint, Metric, analyze};

fn main() {
    // Create action points representing agent behavior over time
    let actions = vec![
        ActionPoint::new("agent-1", 0.0, vec![0.0, 0.0]),
        ActionPoint::new("agent-1", 1.0, vec![1.0, 0.5]),
        ActionPoint::new("agent-1", 2.0, vec![0.5, 1.0]),
        ActionPoint::new("agent-1", 3.0, vec![2.0, 1.5]),
        ActionPoint::new("agent-1", 4.0, vec![1.0, 2.0]),
    ];

    println!("Analyzing {} action points...", actions.len());

    // Run the full persistence pipeline
    let (barcodes, features) = analyze(actions.clone(), Metric::Euclidean, 2);

    // Display barcodes
    println!("\nPersistence Barcodes:");
    for bc in &barcodes.barcodes {
        println!("  H{}: {} bars", bc.dimension, bc.num_bars());
        for (birth, death) in &bc.bars {
            if death.is_infinite() {
                println!("    [{:.2}, ∞)", birth);
            } else {
                println!("    [{:.2}, {:.2})", birth, death);
            }
        }
    }

    // Betti numbers at various scales
    println!("\nBetti numbers at ε=0.5: {:?}", barcodes.betti_numbers(0.5));
    println!("Betti numbers at ε=1.0: {:?}", barcodes.betti_numbers(1.0));
    println!("Betti numbers at ε=2.0: {:?}", barcodes.betti_numbers(2.0));

    // Agent features
    println!("\nAgent Features:");
    println!("  Stability:     {:.3}", features.stability);
    println!("  Adaptability:  {:.3}", features.adaptability);
    println!("  Depth:         {:.3}", features.depth);
    println!("  Archetype:     {}", features.archetype());
    println!("  Personality:   {:?}", features.personality_vector);
}
