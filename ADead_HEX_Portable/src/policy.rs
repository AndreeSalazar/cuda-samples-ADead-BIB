// ADead-BIB HEX - Execution Policy Engine
// "Above CUDA, below frameworks, next to the runtime."
//
// Configurable policies for GPU execution decisions

use std::fs;
use std::path::Path;

/// Execution Policy - Configurable rules for GPU decisions
#[derive(Debug, Clone)]
pub struct ExecutionPolicy {
    /// Policy name
    pub name: String,
    /// Minimum elements for GPU consideration
    pub min_elements: usize,
    /// Minimum FLOPs/Byte ratio
    pub min_flops_per_byte: f64,
    /// Require data persistence for GPU
    pub require_persistence: bool,
    /// Maximum PCIe overhead percentage allowed
    pub max_pcie_overhead_percent: f64,
    /// Enable strict mode (reject all borderline cases)
    pub strict_mode: bool,
    /// Power-aware mode (prefer CPU for power savings)
    pub power_aware: bool,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            min_elements: 100_000,
            min_flops_per_byte: 0.5,
            require_persistence: false,
            max_pcie_overhead_percent: 50.0,
            strict_mode: false,
            power_aware: false,
        }
    }
}

impl ExecutionPolicy {
    /// Create a new policy with a name
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Production policy - conservative, safe
    pub fn production() -> Self {
        Self {
            name: "production".to_string(),
            min_elements: 100_000,
            min_flops_per_byte: 0.5,
            require_persistence: true,
            max_pcie_overhead_percent: 30.0,
            strict_mode: true,
            power_aware: false,
        }
    }

    /// Development policy - permissive, for testing
    pub fn development() -> Self {
        Self {
            name: "development".to_string(),
            min_elements: 10_000,
            min_flops_per_byte: 0.1,
            require_persistence: false,
            max_pcie_overhead_percent: 80.0,
            strict_mode: false,
            power_aware: false,
        }
    }

    /// Edge/Mobile policy - power-conscious
    pub fn edge() -> Self {
        Self {
            name: "edge".to_string(),
            min_elements: 500_000,
            min_flops_per_byte: 1.0,
            require_persistence: true,
            max_pcie_overhead_percent: 20.0,
            strict_mode: true,
            power_aware: true,
        }
    }

    /// Datacenter policy - throughput-focused
    pub fn datacenter() -> Self {
        Self {
            name: "datacenter".to_string(),
            min_elements: 50_000,
            min_flops_per_byte: 0.3,
            require_persistence: true,
            max_pcie_overhead_percent: 40.0,
            strict_mode: false,
            power_aware: false,
        }
    }

    /// Load policy from YAML-like string
    pub fn from_yaml(content: &str) -> Result<Self, String> {
        let mut policy = Self::default();
        
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim();
                let value = value.trim();
                
                match key {
                    "name" => policy.name = value.to_string(),
                    "min_elements" => {
                        policy.min_elements = value.parse()
                            .map_err(|_| format!("Invalid min_elements: {}", value))?;
                    }
                    "min_flops_per_byte" => {
                        policy.min_flops_per_byte = value.parse()
                            .map_err(|_| format!("Invalid min_flops_per_byte: {}", value))?;
                    }
                    "require_persistence" => {
                        policy.require_persistence = value == "true";
                    }
                    "max_pcie_overhead_percent" => {
                        policy.max_pcie_overhead_percent = value.parse()
                            .map_err(|_| format!("Invalid max_pcie_overhead: {}", value))?;
                    }
                    "strict_mode" => {
                        policy.strict_mode = value == "true";
                    }
                    "power_aware" => {
                        policy.power_aware = value == "true";
                    }
                    _ => {} // Ignore unknown keys
                }
            }
        }
        
        Ok(policy)
    }

    /// Load policy from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read policy file: {}", e))?;
        Self::from_yaml(&content)
    }

    /// Save policy to YAML format
    pub fn to_yaml(&self) -> String {
        format!(
r#"# ADead-BIB HEX Execution Policy
# "Above CUDA, below frameworks, next to the runtime."

name: {}
min_elements: {}
min_flops_per_byte: {}
require_persistence: {}
max_pcie_overhead_percent: {}
strict_mode: {}
power_aware: {}
"#,
            self.name,
            self.min_elements,
            self.min_flops_per_byte,
            self.require_persistence,
            self.max_pcie_overhead_percent,
            self.strict_mode,
            self.power_aware
        )
    }

    /// Print policy summary
    pub fn print(&self) {
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  EXECUTION POLICY LOADED                                     ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Name: {:<54}║", self.name);
        println!("║                                                              ║");
        println!("║  GPU Thresholds:                                             ║");
        println!("║    Min elements:        {:>10}                          ║", self.min_elements);
        println!("║    Min FLOPs/Byte:      {:>10.2}                          ║", self.min_flops_per_byte);
        println!("║    Max PCIe overhead:   {:>10.0}%                         ║", self.max_pcie_overhead_percent);
        println!("║                                                              ║");
        println!("║  Modes:                                                      ║");
        println!("║    Require persistence: {:>10}                          ║", if self.require_persistence { "YES" } else { "NO" });
        println!("║    Strict mode:         {:>10}                          ║", if self.strict_mode { "YES" } else { "NO" });
        println!("║    Power-aware:         {:>10}                          ║", if self.power_aware { "YES" } else { "NO" });
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}

/// Policy Engine - Manages and applies execution policies
pub struct PolicyEngine {
    current_policy: ExecutionPolicy,
}

impl PolicyEngine {
    pub fn new() -> Self {
        Self {
            current_policy: ExecutionPolicy::default(),
        }
    }

    pub fn with_policy(policy: ExecutionPolicy) -> Self {
        Self {
            current_policy: policy,
        }
    }

    pub fn load_policy(&mut self, policy: ExecutionPolicy) {
        println!("  Policy loaded: {}", policy.name);
        self.current_policy = policy;
    }

    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), String> {
        let policy = ExecutionPolicy::load_from_file(path)?;
        self.load_policy(policy);
        Ok(())
    }

    pub fn policy(&self) -> &ExecutionPolicy {
        &self.current_policy
    }

    pub fn min_elements(&self) -> usize {
        self.current_policy.min_elements
    }

    pub fn min_flops_per_byte(&self) -> f64 {
        self.current_policy.min_flops_per_byte
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}
