// ADead-BIB HEX - Full Demo with Decision Contracts and Waste Proof
// "CUDA gives power. ADead-BIB gives judgment."
// "The hardware doesn't fail. Decisions do."

use adead_hex_gpu_governor::{
    GpuDispatcher, 
    GpuMisuseDetector,
    MisuseScore,
    DataLocation, 
    operations,
};

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ADead-BIB HEX - GPU Governor Full Demo                      ║");
    println!("║  \"CUDA gives power. ADead-BIB gives judgment.\"               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut dispatcher = GpuDispatcher::new();

    // ========================================================================
    // DEMO 1: Decision Contract for Small Data
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEMO 1: Decision Contract - Small Data (10K elements)");
    println!("═══════════════════════════════════════════════════════════════");
    
    let cost1 = operations::vector_add(10_000, DataLocation::Host, false);
    let contract1 = dispatcher.decide_with_contract(&cost1);
    contract1.print();

    // ========================================================================
    // DEMO 2: GPU Waste Proof
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEMO 2: GPU Waste Proof - Proving the Decision");
    println!("═══════════════════════════════════════════════════════════════");
    
    let proof1 = dispatcher.prove_decision(&cost1);
    proof1.print();

    // ========================================================================
    // DEMO 3: Misuse Score
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEMO 3: GPU Misuse Score");
    println!("═══════════════════════════════════════════════════════════════");
    
    let score1 = MisuseScore::calculate(&cost1);
    score1.print("VectorAdd 10K");

    // ========================================================================
    // DEMO 4: Decision Contract for Large Persistent Data
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEMO 4: Decision Contract - Large Persistent Data (1M)");
    println!("═══════════════════════════════════════════════════════════════");
    
    let cost2 = operations::vector_add(1_000_000, DataLocation::Host, true);
    let contract2 = dispatcher.decide_with_contract(&cost2);
    contract2.print();

    // ========================================================================
    // DEMO 5: MatMul - High Intensity Operation
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEMO 5: MatMul 512x512 - High Compute Intensity");
    println!("═══════════════════════════════════════════════════════════════");
    
    let cost3 = operations::matmul(512, DataLocation::Host, true);
    println!("  Elements: {} (512x512)", 512*512);
    println!("  FLOPs/Byte: {:.2}", cost3.flops_per_byte());
    let contract3 = dispatcher.decide_with_contract(&cost3);
    contract3.print();
    
    let proof3 = dispatcher.prove_decision(&cost3);
    proof3.print();

    // ========================================================================
    // DEMO 6: Data Already on GPU
    // ========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEMO 6: Data Already on GPU - Zero Transfer");
    println!("═══════════════════════════════════════════════════════════════");
    
    let cost4 = operations::vector_add(10_000, DataLocation::Device, false);
    let contract4 = dispatcher.decide_with_contract(&cost4);
    contract4.print();

    // ========================================================================
    // FINAL MESSAGE
    // ========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                                                              ║");
    println!("║  \"Most GPU slowdowns are decision bugs, not hardware bugs.\" ║");
    println!("║                                                              ║");
    println!("║  ADead-BIB HEX: The GPU Governor                             ║");
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
