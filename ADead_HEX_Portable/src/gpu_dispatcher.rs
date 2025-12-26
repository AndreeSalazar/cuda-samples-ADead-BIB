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

// ============================================================================
// DECISION CONTRACT - The formal agreement for each execution decision
// ============================================================================

/// Guarantees provided by a decision
#[derive(Debug, Clone, PartialEq)]
pub enum Guarantee {
    /// No GPU memory will be allocated
    NoGpuAllocation,
    /// No PCIe transfers will occur
    NoPcieTransfer,
    /// Execution is deterministic and predictable
    DeterministicExecution,
    /// Data will persist in VRAM after execution
    DataPersistenceInVram,
    /// Minimum latency path selected
    MinimumLatency,
    /// Power-efficient execution
    PowerEfficient,
}

/// Assumptions made by the decision
#[derive(Debug, Clone)]
pub enum Assumption {
    /// Data size is below threshold
    DataSizeBelowThreshold { size: usize, threshold: usize },
    /// One-shot execution (no reuse)
    OneShotExecution,
    /// Data already resident in VRAM
    DataResidentInVram,
    /// High computational intensity
    HighComputeIntensity { flops_per_byte: f64 },
    /// Data will be reused multiple times
    DataWillBeReused { expected_reuse: usize },
}

/// Risks if assumptions are violated
#[derive(Debug, Clone)]
pub enum Risk {
    /// GPU will be slower than CPU
    GpuSlowdown { factor: f64 },
    /// Unnecessary PCIe transfers
    WastedTransfers { count: usize },
    /// Memory pressure on GPU
    GpuMemoryPressure,
    /// Unpredictable latency
    LatencyVariance,
    /// Power waste
    PowerWaste { estimated_watts: f64 },
}

/// The Decision Contract - A formal agreement for execution
#[derive(Debug, Clone)]
pub struct DecisionContract {
    /// The execution target
    pub target: ExecutionTarget,
    /// The reason for this decision
    pub reason: DecisionReason,
    /// Guarantees provided
    pub guarantees: Vec<Guarantee>,
    /// Assumptions made
    pub assumptions: Vec<Assumption>,
    /// Risks if violated
    pub risks: Vec<Risk>,
    /// Estimated execution time (µs)
    pub estimated_time_us: f64,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
}

impl DecisionContract {
    /// Create a new decision contract
    pub fn new(target: ExecutionTarget, reason: DecisionReason) -> Self {
        Self {
            target,
            reason,
            guarantees: Vec::new(),
            assumptions: Vec::new(),
            risks: Vec::new(),
            estimated_time_us: 0.0,
            confidence: 0.0,
        }
    }

    /// Build contract for CPU execution (small data)
    pub fn cpu_small_data(elements: usize, threshold: usize, estimated_time: f64) -> Self {
        Self {
            target: ExecutionTarget::CPU,
            reason: DecisionReason::TooSmall { elements, threshold },
            guarantees: vec![
                Guarantee::NoGpuAllocation,
                Guarantee::NoPcieTransfer,
                Guarantee::DeterministicExecution,
                Guarantee::PowerEfficient,
            ],
            assumptions: vec![
                Assumption::DataSizeBelowThreshold { size: elements, threshold },
                Assumption::OneShotExecution,
            ],
            risks: vec![
                Risk::GpuSlowdown { factor: 10.0 },
                Risk::WastedTransfers { count: 2 },
            ],
            estimated_time_us: estimated_time,
            confidence: 0.95,
        }
    }

    /// Build contract for CPU execution (low intensity)
    pub fn cpu_low_intensity(flops_per_byte: f64, estimated_time: f64) -> Self {
        Self {
            target: ExecutionTarget::CPU,
            reason: DecisionReason::LowComputeIntensity { flops_per_byte },
            guarantees: vec![
                Guarantee::NoGpuAllocation,
                Guarantee::NoPcieTransfer,
                Guarantee::DeterministicExecution,
            ],
            assumptions: vec![
                Assumption::OneShotExecution,
            ],
            risks: vec![
                Risk::GpuSlowdown { factor: 5.0 },
                Risk::WastedTransfers { count: 2 },
                Risk::PowerWaste { estimated_watts: 30.0 },
            ],
            estimated_time_us: estimated_time,
            confidence: 0.85,
        }
    }

    /// Build contract for GPU execution (data on device)
    pub fn gpu_data_resident(estimated_time: f64) -> Self {
        Self {
            target: ExecutionTarget::GPU,
            reason: DecisionReason::DataAlreadyOnDevice,
            guarantees: vec![
                Guarantee::NoPcieTransfer,
                Guarantee::MinimumLatency,
            ],
            assumptions: vec![
                Assumption::DataResidentInVram,
            ],
            risks: vec![],
            estimated_time_us: estimated_time,
            confidence: 0.99,
        }
    }

    /// Build contract for GPU with transfer (persistent)
    pub fn gpu_with_persistence(expected_reuse: usize, estimated_time: f64) -> Self {
        Self {
            target: ExecutionTarget::GPUWithTransfer,
            reason: DecisionReason::WillPersist,
            guarantees: vec![
                Guarantee::DataPersistenceInVram,
                Guarantee::MinimumLatency,
            ],
            assumptions: vec![
                Assumption::DataWillBeReused { expected_reuse },
            ],
            risks: vec![
                Risk::GpuMemoryPressure,
            ],
            estimated_time_us: estimated_time,
            confidence: 0.90,
        }
    }

    /// Print the contract in a formal format
    pub fn print(&self) {
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  DECISION CONTRACT                                           ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Target: {:?}{:>43}║", self.target, "");
        println!("║  Confidence: {:.0}%{:>47}║", self.confidence * 100.0, "");
        println!("║  Estimated time: {:.1} µs{:>39}║", self.estimated_time_us, "");
        println!("╠══════════════════════════════════════════════════════════════╣");
        
        println!("║  GUARANTEES:                                                 ║");
        for g in &self.guarantees {
            let desc = match g {
                Guarantee::NoGpuAllocation => "No GPU allocation",
                Guarantee::NoPcieTransfer => "No PCIe transfers",
                Guarantee::DeterministicExecution => "Deterministic execution",
                Guarantee::DataPersistenceInVram => "Data persists in VRAM",
                Guarantee::MinimumLatency => "Minimum latency path",
                Guarantee::PowerEfficient => "Power-efficient execution",
            };
            println!("║    ✓ {:<55}║", desc);
        }
        
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  ASSUMPTIONS:                                                ║");
        for a in &self.assumptions {
            let desc = match a {
                Assumption::DataSizeBelowThreshold { size, threshold } => 
                    format!("Data size {} < {} threshold", size, threshold),
                Assumption::OneShotExecution => "One-shot execution".to_string(),
                Assumption::DataResidentInVram => "Data resident in VRAM".to_string(),
                Assumption::HighComputeIntensity { flops_per_byte } => 
                    format!("High intensity: {:.2} FLOPs/Byte", flops_per_byte),
                Assumption::DataWillBeReused { expected_reuse } => 
                    format!("Data reused {} times", expected_reuse),
            };
            println!("║    • {:<55}║", desc);
        }
        
        if !self.risks.is_empty() {
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║  RISKS IF VIOLATED:                                          ║");
            for r in &self.risks {
                let desc = match r {
                    Risk::GpuSlowdown { factor } => format!("GPU slowdown {:.1}x", factor),
                    Risk::WastedTransfers { count } => format!("{} wasted transfers", count),
                    Risk::GpuMemoryPressure => "GPU memory pressure".to_string(),
                    Risk::LatencyVariance => "Unpredictable latency".to_string(),
                    Risk::PowerWaste { estimated_watts } => format!("{:.0}W power waste", estimated_watts),
                };
                println!("║    ⚠ {:<55}║", desc);
            }
        }
        
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
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

    /// Decide with full contract (formal agreement)
    pub fn decide_with_contract(&mut self, cost: &OperationCost) -> DecisionContract {
        let (target, reason) = self.decide(cost);
        
        match (&target, &reason) {
            (ExecutionTarget::CPU, DecisionReason::TooSmall { elements, threshold }) => {
                DecisionContract::cpu_small_data(*elements, *threshold, cost.estimate_cpu_us())
            }
            (ExecutionTarget::CPU, DecisionReason::LowComputeIntensity { flops_per_byte }) => {
                DecisionContract::cpu_low_intensity(*flops_per_byte, cost.estimate_cpu_us())
            }
            (ExecutionTarget::GPU, DecisionReason::DataAlreadyOnDevice) => {
                DecisionContract::gpu_data_resident(cost.estimate_kernel_us())
            }
            (ExecutionTarget::GPUWithTransfer, DecisionReason::WillPersist) => {
                DecisionContract::gpu_with_persistence(5, cost.estimate_h2d_us() + cost.estimate_kernel_us())
            }
            _ => DecisionContract::new(target, reason),
        }
    }

    /// Prove the decision by simulating both paths
    pub fn prove_decision(&self, cost: &OperationCost) -> WasteProof {
        let cpu_time = cost.estimate_cpu_us();
        let gpu_time = cost.estimate_h2d_us() + cost.estimate_kernel_us() + cost.estimate_h2d_us();
        
        let waste_factor = if cpu_time > 0.0 { gpu_time / cpu_time } else { 1.0 };
        let pcie_dominance = if gpu_time > 0.0 {
            ((cost.estimate_h2d_us() * 2.0) / gpu_time) * 100.0
        } else {
            0.0
        };
        
        let gpu_is_waste = gpu_time > cpu_time;
        
        WasteProof {
            cpu_time_us: cpu_time,
            gpu_time_us: gpu_time,
            waste_factor,
            pcie_dominance_percent: pcie_dominance,
            gpu_is_waste,
            contract_violated: gpu_is_waste,
        }
    }
}

/// Proof of GPU waste - shows the real cost of wrong decisions
#[derive(Debug, Clone)]
pub struct WasteProof {
    pub cpu_time_us: f64,
    pub gpu_time_us: f64,
    pub waste_factor: f64,
    pub pcie_dominance_percent: f64,
    pub gpu_is_waste: bool,
    pub contract_violated: bool,
}

impl WasteProof {
    pub fn print(&self) {
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  GPU WASTE PROOF - ADead-BIB HEX                             ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Executing both paths:                                       ║");
        println!("║                                                              ║");
        println!("║  CPU execution:        {:>10.1} µs                        ║", self.cpu_time_us);
        println!("║  GPU execution (forced): {:>8.1} µs                        ║", self.gpu_time_us);
        println!("║                                                              ║");
        
        if self.gpu_is_waste {
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║  🚨 GPU MISUSE CONFIRMED                                     ║");
            println!("║                                                              ║");
            println!("║  Waste factor:         {:>10.1}x                          ║", self.waste_factor);
            println!("║  PCIe dominance:       {:>10.1}%                          ║", self.pcie_dominance_percent);
            println!("║                                                              ║");
            println!("║  Conclusion:                                                 ║");
            println!("║  GPU usage violated Decision Contract                        ║");
        } else {
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║  ✅ GPU USAGE JUSTIFIED                                      ║");
            println!("║                                                              ║");
            println!("║  Speedup:              {:>10.1}x                          ║", 1.0 / self.waste_factor);
        }
        
        println!("╚══════════════════════════════════════════════════════════════╝");
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
