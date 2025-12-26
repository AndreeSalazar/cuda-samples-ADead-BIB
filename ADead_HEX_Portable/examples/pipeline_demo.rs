// ADead-BIB HEX - Pipeline Demo
// Shows real pipeline optimization: Preprocess → VectorAdd → Reduce → Postprocess
// "CUDA gives power. ADead-BIB gives judgment."

use adead_hex_gpu_governor::{
    GpuDispatcher, 
    DataLocation, 
    operations,
    ExecutionTarget,
};

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ADead-BIB HEX - Pipeline Demo                               ║");
    println!("║  Real Pipeline: Preprocess → Compute → Reduce → Postprocess  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut dispatcher = GpuDispatcher::new();
    let n = 500_000; // 500K elements

    // ========================================================================
    // SCENARIO A: CUDA Naive (Always GPU, transfer every time)
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SCENARIO A: CUDA Naive (Always GPU)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut naive_total_time = 0.0;
    let mut naive_transfers = 0;

    // Step 1: Preprocess (small, should be CPU)
    let preprocess = operations::vector_add(10_000, DataLocation::Host, false);
    naive_total_time += preprocess.estimate_h2d_us() * 2.0 + preprocess.estimate_kernel_us();
    naive_transfers += 2;
    println!("  Step 1: Preprocess (10K) → GPU forced");
    println!("    Time: {:.1} µs, Transfers: 2", preprocess.estimate_h2d_us() * 2.0 + preprocess.estimate_kernel_us());

    // Step 2: VectorAdd (large)
    let vectoradd = operations::vector_add(n, DataLocation::Host, false);
    naive_total_time += vectoradd.estimate_h2d_us() * 2.0 + vectoradd.estimate_kernel_us();
    naive_transfers += 2;
    println!("  Step 2: VectorAdd (500K) → GPU forced");
    println!("    Time: {:.1} µs, Transfers: 2", vectoradd.estimate_h2d_us() * 2.0 + vectoradd.estimate_kernel_us());

    // Step 3: SAXPY
    let saxpy = operations::saxpy(n, DataLocation::Host, false);
    naive_total_time += saxpy.estimate_h2d_us() * 2.0 + saxpy.estimate_kernel_us();
    naive_transfers += 2;
    println!("  Step 3: SAXPY (500K) → GPU forced");
    println!("    Time: {:.1} µs, Transfers: 2", saxpy.estimate_h2d_us() * 2.0 + saxpy.estimate_kernel_us());

    // Step 4: Reduce
    let reduce = operations::reduction(n, DataLocation::Host);
    naive_total_time += reduce.estimate_h2d_us() * 2.0 + reduce.estimate_kernel_us();
    naive_transfers += 2;
    println!("  Step 4: Reduce (500K) → GPU forced");
    println!("    Time: {:.1} µs, Transfers: 2", reduce.estimate_h2d_us() * 2.0 + reduce.estimate_kernel_us());

    // Step 5: Postprocess (small)
    let postprocess = operations::vector_add(5_000, DataLocation::Host, false);
    naive_total_time += postprocess.estimate_h2d_us() * 2.0 + postprocess.estimate_kernel_us();
    naive_transfers += 2;
    println!("  Step 5: Postprocess (5K) → GPU forced");
    println!("    Time: {:.1} µs, Transfers: 2", postprocess.estimate_h2d_us() * 2.0 + postprocess.estimate_kernel_us());

    println!();
    println!("  NAIVE TOTAL: {:.1} µs, {} transfers", naive_total_time, naive_transfers);

    // ========================================================================
    // SCENARIO B: ADead-BIB Governor (Smart decisions)
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SCENARIO B: ADead-BIB Governor (Smart Decisions)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut smart_total_time = 0.0;
    let mut smart_transfers = 0;

    // Step 1: Preprocess (small) → CPU
    let preprocess = operations::vector_add(10_000, DataLocation::Host, false);
    let (target1, _) = dispatcher.decide(&preprocess);
    if target1 == ExecutionTarget::CPU {
        smart_total_time += preprocess.estimate_cpu_us();
        println!("  Step 1: Preprocess (10K) → CPU ✓");
        println!("    Time: {:.1} µs, Transfers: 0", preprocess.estimate_cpu_us());
    }

    // Step 2: VectorAdd (large, persist) → GPU with transfer
    let vectoradd = operations::vector_add(n, DataLocation::Host, true);
    let (target2, _) = dispatcher.decide(&vectoradd);
    if matches!(target2, ExecutionTarget::GPUWithTransfer) {
        smart_total_time += vectoradd.estimate_h2d_us() + vectoradd.estimate_kernel_us();
        smart_transfers += 1; // Only H2D, data persists
        println!("  Step 2: VectorAdd (500K) → GPU + Persist ✓");
        println!("    Time: {:.1} µs, Transfers: 1 (H2D only)", vectoradd.estimate_h2d_us() + vectoradd.estimate_kernel_us());
    }

    // Step 3: SAXPY (data already on GPU)
    let saxpy = operations::saxpy(n, DataLocation::Device, true);
    let (target3, _) = dispatcher.decide(&saxpy);
    if target3 == ExecutionTarget::GPU {
        smart_total_time += saxpy.estimate_kernel_us();
        println!("  Step 3: SAXPY (500K) → GPU (data resident) ✓");
        println!("    Time: {:.1} µs, Transfers: 0", saxpy.estimate_kernel_us());
    }

    // Step 4: Reduce (data on GPU)
    let reduce = operations::reduction(n, DataLocation::Device);
    let (target4, _) = dispatcher.decide(&reduce);
    if target4 == ExecutionTarget::GPU {
        smart_total_time += reduce.estimate_kernel_us();
        println!("  Step 4: Reduce (500K) → GPU (data resident) ✓");
        println!("    Time: {:.1} µs, Transfers: 0", reduce.estimate_kernel_us());
    }

    // Step 5: Postprocess (small) → CPU, need D2H
    let postprocess = operations::vector_add(5_000, DataLocation::Host, false);
    let (target5, _) = dispatcher.decide(&postprocess);
    if target5 == ExecutionTarget::CPU {
        // Need to bring data back from GPU first
        let d2h_time = vectoradd.estimate_h2d_us(); // Approximate D2H
        smart_total_time += d2h_time + postprocess.estimate_cpu_us();
        smart_transfers += 1; // D2H
        println!("  Step 5: Postprocess (5K) → CPU (after D2H) ✓");
        println!("    Time: {:.1} µs, Transfers: 1 (D2H)", d2h_time + postprocess.estimate_cpu_us());
    }

    println!();
    println!("  SMART TOTAL: {:.1} µs, {} transfers", smart_total_time, smart_transfers);

    // ========================================================================
    // COMPARISON
    // ========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  COMPARISON                                                  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                              ║");
    println!("║  Scenario A: CUDA Naive                                      ║");
    println!("║    Total time:  {:>10.1} µs                               ║", naive_total_time);
    println!("║    Transfers:   {:>10}                                   ║", naive_transfers);
    println!("║                                                              ║");
    println!("║  Scenario B: ADead-BIB Governor                              ║");
    println!("║    Total time:  {:>10.1} µs                               ║", smart_total_time);
    println!("║    Transfers:   {:>10}                                   ║", smart_transfers);
    println!("║                                                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    
    let speedup = naive_total_time / smart_total_time;
    let transfer_reduction = ((naive_transfers - smart_transfers) as f64 / naive_transfers as f64) * 100.0;
    
    println!("║  🔥 EFFICIENCY GAIN: {:.1}x faster                           ║", speedup);
    println!("║  🔥 TRANSFER REDUCTION: {:.0}%                               ║", transfer_reduction);
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!();
    println!("  \"This is SYSTEM optimization, not benchmark optimization.\"");
    println!();
}
