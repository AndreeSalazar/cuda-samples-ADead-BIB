# 🔥 ADead-BIB HEX: The GPU Governor

> **"CUDA gives power. ADead-BIB gives judgment."**
> **"The hardware doesn't fail. Decisions do."**

## What Is This?

A **standalone Rust library** that governs GPU execution decisions. It prevents GPU misuse by deciding **when** to use GPU vs CPU based on a cost model.

This is a **portable version** extracted from the main ADead-BIB project.

---

## Quick Start

```rust
use adead_hex_gpu_governor::{GpuDispatcher, DataLocation, operations};

fn main() {
    let mut dispatcher = GpuDispatcher::new();
    
    // Small data → CPU
    let cost = operations::vector_add(10_000, DataLocation::Host, false);
    let (target, reason) = dispatcher.decide(&cost);
    println!("Decision: {:?}", target); // CPU
    
    // Large data with persistence → GPU
    let cost = operations::matmul(512, DataLocation::Host, true);
    let (target, reason) = dispatcher.decide(&cost);
    println!("Decision: {:?}", target); // GPUWithTransfer
}
```

---

## Features

### 1. GPU Dispatcher

Automatic CPU↔GPU decisions based on:
- Element count (threshold: 100K)
- FLOPs/Byte ratio (threshold: 0.5)
- Data location (Host vs Device)
- Persistence (will data be reused?)

### 2. GPU Misuse Detector

Detects and reports incorrect GPU usage:
- Kernel too small
- Low computational intensity
- Unnecessary transfers

### 3. Misuse Score (0-100)

Quantifiable metric for GPU misuse:

```
╔══════════════════════════════════════════════════════════════╗
║  GPU MISUSE SCORE: 94 / 100 (CRITICAL)                       ║
╠══════════════════════════════════════════════════════════════╣
║  Breakdown:                                                  ║
║  ├── PCIe overhead dominance:     +40 points                ║
║  ├── Low arithmetic intensity:    +25 points                ║
║  ├── One-shot execution:          +15 points                ║
║  └── Small element count:         +4 points                 ║
║                                                              ║
║  Recommendation: Execute on CPU                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Cost Model

Based on real RTX 3060 benchmarks:

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `GPU_THRESHOLD_ELEMENTS` | 100,000 | Min elements for GPU |
| `MIN_FLOPS_PER_BYTE` | 0.5 | Min compute intensity |
| `PCIE_TRANSFER_THRESHOLD` | 10 MB | Consider persistence |

---

## Decision Logic

```rust
fn decide(operation) -> Target {
    if data_on_device { return GPU }
    if elements < 100K { return CPU }
    if flops_per_byte > 0.5 { return GPU }
    if will_persist { return GPUWithTransfer }
    if gpu_time < cpu_time { return GPU }
    else { return CPU }
}
```

---

## Run Demo

```bash
cd ADead_HEX_Portable
cargo run --example demo
```

---

## Files

```
ADead_HEX_Portable/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── gpu_dispatcher.rs
│   └── gpu_misuse_detector.rs
├── examples/
│   └── demo.rs
└── docs/
    └── (documentation)
```

---

## Why This Matters

| Problem | CUDA Alone | With ADead-BIB HEX |
|---------|------------|-------------------|
| Small kernels | Slow (PCIe overhead) | Rejected → CPU |
| One-shot data | Transfers dominate | Detected, warned |
| Low intensity | Wasted GPU cycles | CPU preferred |
| Misuse detection | None | Score 0-100 |

---

## License

Apache 2.0

---

## Author

**Eddi Andreé Salazar Matos**  
eddi.salazar.dev@gmail.com

---

*ADead-BIB HEX - The GPU Governor*
