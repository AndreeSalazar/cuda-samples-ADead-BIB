// ADead-BIB HEX - GPU Dispatcher (Portable Version)
// Host Determinista que Gobierna Ejecución GPU
// Cost Model: bytes vs FLOPs → decisión automática CPU↔GPU
//
// This is a standalone portable version that can be used independently
// of the main ADead-BIB project.

/// Minimum elements threshold for GPU consideration
/// Based on real RTX 3060 benchmarks:
/// - < 100K: CPU wins (PCIe overhead)
/// - > 100K: GPU kernel wins
/// - But transfers dominate if data doesn't persist
pub const GPU_THRESHOLD_ELEMENTS: usize = 100_000;

/// PCIe transfer threshold in bytes
/// PCIe 3.0 x16: ~12 GB/s theoretical, ~10 GB/s real
pub const PCIE_TRANSFER_THRESHOLD_BYTES: usize = 10_000_000; // 10 MB

/// Minimum FLOPs/Byte ratio to justify GPU
/// If few FLOPs per byte transferred, CPU wins
/// Example: VectorAdd = 1 FLOP / 12 bytes = 0.08 (very low)
/// MatMul NxN = 2N³ FLOPs / 3N² bytes = 0.67N (scales with N)
pub const MIN_FLOPS_PER_BYTE: f64 = 0.5;

/// Data location in the system
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataLocation {
    /// Data in host RAM
    Host,
    /// Data in GPU VRAM
    Device,
    /// Data in both (synchronized)
    Both,
    /// Data in GPU, host outdated
    DeviceDirty,
}

/// Dispatcher decision
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionTarget {
    /// Execute on CPU
    CPU,
    /// Execute on GPU (data already in VRAM)
    GPU,
    /// Transfer to GPU, execute, keep in VRAM
    GPUWithTransfer,
    /// Transfer to GPU, execute, bring back
    GPURoundTrip,
}

/// Decision reason (for debugging/logging)
#[derive(Debug, Clone)]
pub enum DecisionReason {
    /// Data too small, overhead dominates
    TooSmall { elements: usize, threshold: usize },
    /// Data already on GPU, execute there
    DataAlreadyOnDevice,
    /// Enough FLOPs to justify transfer
    HighComputeIntensity { flops_per_byte: f64 },
    /// Few FLOPs, transfer not worth it
    LowComputeIntensity { flops_per_byte: f64 },
    /// Data will persist, worth transferring
    WillPersist,
    /// GPU not available
    NoGPU,
}

/// Cost Model for operations
#[derive(Debug, Clone)]
pub struct OperationCost {
    /// Operation name
    pub name: String,
    /// Number of elements
    pub elements: usize,
    /// Bytes per element
    pub bytes_per_element: usize,
    /// FLOPs per element
    pub flops_per_element: usize,
    /// Will data persist in GPU after?
    pub will_persist: bool,
    /// Current data location
    pub data_location: DataLocation,
}

impl OperationCost {
    /// Calculate total bytes
    pub fn total_bytes(&self) -> usize {
        self.elements * self.bytes_per_element
    }

    /// Calculate total FLOPs
    pub fn total_flops(&self) -> usize {
        self.elements * self.flops_per_element
    }

    /// Calculate FLOPs/Byte ratio
    pub fn flops_per_byte(&self) -> f64 {
        if self.bytes_per_element == 0 {
            return 0.0;
        }
        self.flops_per_element as f64 / self.bytes_per_element as f64
    }

    /// Estimate H2D transfer time in microseconds
    /// Based on PCIe 3.0 x16: ~10 GB/s
    pub fn estimate_h2d_us(&self) -> f64 {
        let bytes = self.total_bytes() as f64;
        let bandwidth = 10_000_000_000.0; // 10 GB/s
        (bytes / bandwidth) * 1_000_000.0
    }

    /// Estimate GPU kernel time in microseconds
    /// Based on RTX 3060 benchmark: ~300 GFLOPS for simple ops
    pub fn estimate_kernel_us(&self) -> f64 {
        let flops = self.total_flops() as f64;
        let throughput = 300_000_000_000.0; // 300 GFLOPS conservative
        (flops / throughput) * 1_000_000.0
    }

    /// Estimate CPU time in microseconds
    /// Based on benchmark: ~1 GFLOP/s for simple ops
    pub fn estimate_cpu_us(&self) -> f64 {
        let flops = self.total_flops() as f64;
        let throughput = 1_000_000_000.0; // 1 GFLOPS conservative
        (flops / throughput) * 1_000_000.0
    }
}

/// GPU Dispatcher - The brain of ADead-BIB HEX
pub struct GpuDispatcher {
    /// GPU available?
    gpu_available: bool,
    /// Element threshold
    threshold_elements: usize,
    /// Decision history (for future learning)
    decision_history: Vec<(OperationCost, ExecutionTarget, DecisionReason)>,
}

impl GpuDispatcher {
    pub fn new() -> Self {
        Self {
            gpu_available: true, // Assume GPU available
            threshold_elements: GPU_THRESHOLD_ELEMENTS,
            decision_history: Vec::new(),
        }
    }

    /// Decide where to execute an operation
    pub fn decide(&mut self, cost: &OperationCost) -> (ExecutionTarget, DecisionReason) {
        // 1. GPU available?
        if !self.gpu_available {
            return (ExecutionTarget::CPU, DecisionReason::NoGPU);
        }

        // 2. Data already on GPU?
        if cost.data_location == DataLocation::Device 
           || cost.data_location == DataLocation::Both {
            return (ExecutionTarget::GPU, DecisionReason::DataAlreadyOnDevice);
        }

        // 3. Enough elements?
        if cost.elements < self.threshold_elements {
            return (
                ExecutionTarget::CPU,
                DecisionReason::TooSmall {
                    elements: cost.elements,
                    threshold: self.threshold_elements,
                },
            );
        }

        // 4. Enough computational intensity?
        let fpb = cost.flops_per_byte();
        if fpb < MIN_FLOPS_PER_BYTE && !cost.will_persist {
            return (
                ExecutionTarget::CPU,
                DecisionReason::LowComputeIntensity { flops_per_byte: fpb },
            );
        }

        // 5. Will data persist?
        if cost.will_persist {
            return (ExecutionTarget::GPUWithTransfer, DecisionReason::WillPersist);
        }

        // 6. Compare estimated times
        let cpu_time = cost.estimate_cpu_us();
        let gpu_time = cost.estimate_h2d_us() * 2.0 + cost.estimate_kernel_us();

        if gpu_time < cpu_time {
            (
                ExecutionTarget::GPURoundTrip,
                DecisionReason::HighComputeIntensity { flops_per_byte: fpb },
            )
        } else {
            (
                ExecutionTarget::CPU,
                DecisionReason::LowComputeIntensity { flops_per_byte: fpb },
            )
        }
    }

    /// Log a decision for analysis
    pub fn log_decision(&mut self, cost: OperationCost, target: ExecutionTarget, reason: DecisionReason) {
        self.decision_history.push((cost, target, reason));
    }

    /// Print decision summary
    pub fn print_summary(&self) {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  ADead-BIB HEX - GPU Dispatcher Summary                      ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();

        let mut cpu_count = 0;
        let mut gpu_count = 0;

        for (cost, target, reason) in &self.decision_history {
            match target {
                ExecutionTarget::CPU => cpu_count += 1,
                _ => gpu_count += 1,
            }
            println!("  {} ({} elements) → {:?}", cost.name, cost.elements, target);
            println!("    Reason: {:?}", reason);
        }

        println!();
        println!("  Total: {} CPU, {} GPU", cpu_count, gpu_count);
    }
}

impl Default for GpuDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Predefined operations with their cost model
pub mod operations {
    use super::*;

    /// VectorAdd: C = A + B
    /// 1 FLOP per element, 12 bytes per element (3 floats)
    pub fn vector_add(n: usize, location: DataLocation, persist: bool) -> OperationCost {
        OperationCost {
            name: "VectorAdd".to_string(),
            elements: n,
            bytes_per_element: 12,
            flops_per_element: 1,
            will_persist: persist,
            data_location: location,
        }
    }

    /// SAXPY: Y = a*X + Y
    /// 2 FLOPs per element (mul + add), 8 bytes per element (2 floats)
    pub fn saxpy(n: usize, location: DataLocation, persist: bool) -> OperationCost {
        OperationCost {
            name: "SAXPY".to_string(),
            elements: n,
            bytes_per_element: 8,
            flops_per_element: 2,
            will_persist: persist,
            data_location: location,
        }
    }

    /// MatMul: C = A * B (NxN matrices)
    /// 2N FLOPs per element of C, 12 bytes per element
    pub fn matmul(n: usize, location: DataLocation, persist: bool) -> OperationCost {
        OperationCost {
            name: "MatMul".to_string(),
            elements: n * n,
            bytes_per_element: 12,
            flops_per_element: 2 * n,
            will_persist: persist,
            data_location: location,
        }
    }

    /// Reduction: sum(A)
    /// 1 FLOP per element, 4 bytes per element
    pub fn reduction(n: usize, location: DataLocation) -> OperationCost {
        OperationCost {
            name: "Reduction".to_string(),
            elements: n,
            bytes_per_element: 4,
            flops_per_element: 1,
            will_persist: false,
            data_location: location,
        }
    }
}
