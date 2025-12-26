// ADead-BIB HEX - GPU Governor Demo
// Shows how the dispatcher makes intelligent CPU↔GPU decisions

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
    println!("║  ADead-BIB HEX - GPU Governor Demo                           ║");
    println!("║  \"CUDA gives power. ADead-BIB gives judgment.\"               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut dispatcher = GpuDispatcher::new();
    let mut detector = GpuMisuseDetector::new();

    // Scenario 1: Small data → CPU
    println!("═══ Scenario 1: VectorAdd 10K elements ═══");
    let cost1 = operations::vector_add(10_000, DataLocation::Host, false);
    let (target1, reason1) = dispatcher.decide(&cost1);
    println!("  Elements: 10,000");
    println!("  Decision: {:?}", target1);
    println!("  Reason: {:?}", reason1);
    
    let score1 = MisuseScore::calculate(&cost1);
    score1.print("VectorAdd 10K");
    println!();

    // Scenario 2: Large data, one-shot → Depends
    println!("═══ Scenario 2: VectorAdd 1M elements (one-shot) ═══");
    let cost2 = operations::vector_add(1_000_000, DataLocation::Host, false);
    let (target2, reason2) = dispatcher.decide(&cost2);
    println!("  Elements: 1,000,000");
    println!("  Decision: {:?}", target2);
    println!("  Reason: {:?}", reason2);
    detector.analyze(&cost2);
    println!();

    // Scenario 3: Large data, persistent → GPU
    println!("═══ Scenario 3: VectorAdd 1M elements (persistent) ═══");
    let cost3 = operations::vector_add(1_000_000, DataLocation::Host, true);
    let (target3, reason3) = dispatcher.decide(&cost3);
    println!("  Elements: 1,000,000");
    println!("  Persistence: YES");
    println!("  Decision: {:?}", target3);
    println!("  Reason: {:?}", reason3);
    println!();

    // Scenario 4: MatMul → GPU always
    println!("═══ Scenario 4: MatMul 512x512 ═══");
    let cost4 = operations::matmul(512, DataLocation::Host, true);
    let (target4, reason4) = dispatcher.decide(&cost4);
    println!("  Elements: {} (512x512)", 512*512);
    println!("  FLOPs/Byte: {:.2}", cost4.flops_per_byte());
    println!("  Decision: {:?}", target4);
    println!("  Reason: {:?}", reason4);
    println!();

    // Scenario 5: Data already on GPU
    println!("═══ Scenario 5: Data already on GPU ═══");
    let cost5 = operations::vector_add(10_000, DataLocation::Device, false);
    let (target5, reason5) = dispatcher.decide(&cost5);
    println!("  Elements: 10,000");
    println!("  Location: Device (VRAM)");
    println!("  Decision: {:?}", target5);
    println!("  Reason: {:?}", reason5);
    println!();

    // Print misuse report
    detector.print_report();

    println!("════════════════════════════════════════════════════════════════");
    println!("  ADead-BIB HEX: The GPU Governor");
    println!("  \"The hardware doesn't fail. Decisions do.\"");
    println!("════════════════════════════════════════════════════════════════");
}
