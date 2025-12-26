// ADead-BIB HEX - GPU Governor (Portable Library)
// 
// "CUDA gives power. ADead-BIB gives judgment."
// "The hardware doesn't fail. Decisions do."
//
// This is a standalone portable version that can be used independently
// of the main ADead-BIB project.

pub mod gpu_dispatcher;
pub mod gpu_misuse_detector;
pub mod policy;

pub use gpu_dispatcher::{
    GpuDispatcher, 
    ExecutionTarget, 
    DataLocation, 
    OperationCost, 
    DecisionReason,
    DecisionContract,
    Guarantee,
    Assumption,
    Risk,
    WasteProof,
    GPU_THRESHOLD_ELEMENTS,
    MIN_FLOPS_PER_BYTE,
    operations,
};

pub use gpu_misuse_detector::{
    GpuMisuseDetector,
    MisuseReport,
    MisuseSeverity,
    MisuseType,
    MisuseScore,
};

pub use policy::{
    ExecutionPolicy,
    PolicyEngine,
};
