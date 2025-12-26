// ADead-BIB HEX - Adversarial Demo
// Shows what happens when developers make ALL the wrong decisions
// "Even when the developer is wrong, the system corrects."

use adead_hex_gpu_governor::{
    GpuDispatcher, 
    GpuMisuseDetector,
    MisuseScore,
    DataLocation, 
    ExecutionTarget,
    OperationCost,
};

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ADead-BIB HEX - Adversarial Demo                            ║");
    println!("║  \"Even when the developer is wrong, the system corrects.\"    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  This demo intentionally makes ALL the wrong decisions:");
    println!("  - Small kernels forced to GPU");
    println!("  - No data persistence");
    println!("  - Unnecessary transfers");
    println!("  - Low compute intensity operations");
    println!();

    // ========================================================================
    // SCENARIO: Developer forces everything to GPU (worst case)
    // ========================================================================
    
    let mut detector = GpuMisuseDetector::new();
    let mut dispatcher = GpuDispatcher::new();

    // Define a series of BAD operations (what a naive developer would do)
    let bad_operations = vec![
        // Small kernel #1
        OperationCost {
            name: "SmallKernel_1".to_string(),
            elements: 5_000,
            bytes_per_element: 12,
            flops_per_element: 1,
            will_persist: false,
            data_location: DataLocation::Host,
        },
        // Small kernel #2
        OperationCost {
            name: "SmallKernel_2".to_string(),
            elements: 8_000,
            bytes_per_element: 8,
            flops_per_element: 2,
            will_persist: false,
            data_location: DataLocation::Host,
        },
        // Medium but low intensity
        OperationCost {
            name: "LowIntensity_1".to_string(),
            elements: 50_000,
            bytes_per_element: 12,
            flops_per_element: 1,
            will_persist: false,
            data_location: DataLocation::Host,
        },
        // Small kernel #3
        OperationCost {
            name: "SmallKernel_3".to_string(),
            elements: 3_000,
            bytes_per_element: 4,
            flops_per_element: 1,
            will_persist: false,
            data_location: DataLocation::Host,
        },
        // Another low intensity
        OperationCost {
            name: "LowIntensity_2".to_string(),
            elements: 40_000,
            bytes_per_element: 16,
            flops_per_element: 1,
            will_persist: false,
            data_location: DataLocation::Host,
        },
        // Tiny kernel
        OperationCost {
            name: "TinyKernel".to_string(),
            elements: 1_000,
            bytes_per_element: 8,
            flops_per_element: 1,
            will_persist: false,
            data_location: DataLocation::Host,
        },
        // One more small
        OperationCost {
            name: "SmallKernel_4".to_string(),
            elements: 7_500,
            bytes_per_element: 12,
            flops_per_element: 1,
            will_persist: false,
            data_location: DataLocation::Host,
        },
    ];

    // ========================================================================
    // WITHOUT POLICY ENGINE (Naive CUDA - force everything to GPU)
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  WITHOUT Policy Engine (Naive CUDA)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut naive_total_time = 0.0;
    let mut naive_transfers = 0;
    let mut total_misuse_score = 0u32;

    for op in &bad_operations {
        // Force GPU execution (what naive developer does)
        let gpu_time = op.estimate_h2d_us() * 2.0 + op.estimate_kernel_us();
        naive_total_time += gpu_time;
        naive_transfers += 2; // H2D + D2H every time
        
        let score = MisuseScore::calculate(op);
        total_misuse_score += score.total;
        
        println!("  {} → GPU forced", op.name);
        println!("    Time: {:.1} µs, Transfers: 2, Misuse: {}/100", gpu_time, score.total);
    }

    let avg_misuse = total_misuse_score / bad_operations.len() as u32;
    
    println!();
    println!("  ┌────────────────────────────────────────────────────────────┐");
    println!("  │  NAIVE TOTAL                                               │");
    println!("  │  Time:      {:>10.1} µs                                 │", naive_total_time);
    println!("  │  Transfers: {:>10}                                     │", naive_transfers);
    println!("  │  Avg Misuse Score: {:>3}/100 (SEVERE)                      │", avg_misuse);
    println!("  └────────────────────────────────────────────────────────────┘");

    // ========================================================================
    // WITH POLICY ENGINE (ADead-BIB HEX)
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  WITH Policy Engine (ADead-BIB HEX)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut smart_total_time = 0.0;
    let mut smart_transfers = 0;
    let mut smart_misuse_score = 0u32;
    let mut cpu_decisions = 0;
    let mut gpu_decisions = 0;

    for op in &bad_operations {
        let (target, _reason) = dispatcher.decide(op);
        
        let time = match target {
            ExecutionTarget::CPU => {
                cpu_decisions += 1;
                op.estimate_cpu_us()
            }
            _ => {
                gpu_decisions += 1;
                smart_transfers += 2;
                op.estimate_h2d_us() * 2.0 + op.estimate_kernel_us()
            }
        };
        
        smart_total_time += time;
        
        // With policy engine, misuse is prevented
        let effective_score = if target == ExecutionTarget::CPU { 0 } else { 
            MisuseScore::calculate(op).total 
        };
        smart_misuse_score += effective_score;
        
        detector.analyze(op);
        
        let target_str = match target {
            ExecutionTarget::CPU => "CPU ✓",
            _ => "GPU",
        };
        
        println!("  {} → {}", op.name, target_str);
        println!("    Time: {:.1} µs, Misuse: {}/100", time, effective_score);
    }

    let smart_avg_misuse = smart_misuse_score / bad_operations.len() as u32;

    println!();
    println!("  ┌────────────────────────────────────────────────────────────┐");
    println!("  │  SMART TOTAL                                               │");
    println!("  │  Time:      {:>10.1} µs                                 │", smart_total_time);
    println!("  │  Transfers: {:>10}                                     │", smart_transfers);
    println!("  │  Avg Misuse Score: {:>3}/100 (SAFE)                        │", smart_avg_misuse);
    println!("  │  Decisions: {} CPU, {} GPU                                 │", cpu_decisions, gpu_decisions);
    println!("  └────────────────────────────────────────────────────────────┘");

    // ========================================================================
    // COMPARISON
    // ========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  COMPARISON                                                  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                              ║");
    println!("║  Without Policy Engine:                                      ║");
    println!("║    Transfers: {:>10}                                     ║", naive_transfers);
    println!("║    Time:      {:>10.1} µs                                 ║", naive_total_time);
    println!("║    Misuse Score: {:>3}/100 (Severe)                          ║", avg_misuse);
    println!("║                                                              ║");
    println!("║  With Policy Engine:                                         ║");
    println!("║    Transfers: {:>10}                                     ║", smart_transfers);
    println!("║    Time:      {:>10.1} µs                                 ║", smart_total_time);
    println!("║    Misuse Score: {:>3}/100 (Safe)                            ║", smart_avg_misuse);
    println!("║                                                              ║");
    
    let speedup = naive_total_time / smart_total_time;
    let transfer_reduction = ((naive_transfers - smart_transfers) as f64 / naive_transfers as f64) * 100.0;
    let misuse_reduction = ((avg_misuse - smart_avg_misuse) as f64 / avg_misuse as f64) * 100.0;
    
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  🔥 SPEEDUP: {:.1}x faster                                    ║", speedup);
    println!("║  🔥 TRANSFER REDUCTION: {:.0}%                                ║", transfer_reduction);
    println!("║  🔥 MISUSE REDUCTION: {:.0}%                                  ║", misuse_reduction);
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!();
    println!("  ┌────────────────────────────────────────────────────────────┐");
    println!("  │                                                            │");
    println!("  │  \"Even when the developer makes ALL the wrong decisions,  │");
    println!("  │   the policy engine corrects them.\"                       │");
    println!("  │                                                            │");
    println!("  │  This is INFRASTRUCTURE, not a demo.                       │");
    println!("  │                                                            │");
    println!("  └────────────────────────────────────────────────────────────┘");
    println!();
    println!("  ADead-BIB HEX - Execution Policy Engine");
    println!("  \"Above CUDA, below frameworks, next to the runtime.\"");
    println!();
}
