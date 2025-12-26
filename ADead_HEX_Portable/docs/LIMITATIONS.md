# Limitations

> **"Knowing what a system does NOT do is as important as knowing what it does."**

---

## What ADead-BIB HEX Does NOT Do

### ❌ Does NOT Replace Kernel Optimization

ADead-BIB operates at the **decision level**, not the kernel level.

- We don't optimize memory coalescing
- We don't tune block/thread configurations
- We don't modify PTX/SASS
- We don't replace cuBLAS, cuDNN, or other optimized libraries

If your kernel is slow, ADead-BIB won't make it faster.
We decide **whether to run it at all**.

---

### ❌ Does NOT Auto-tune Algorithms

There is no machine learning, no adaptive tuning, no magic.

- Thresholds are policy-driven (configurable YAML)
- Decisions are deterministic and auditable
- No hidden heuristics that change over time

This is intentional. Predictability > cleverness.

---

### ❌ Does NOT Profile Execution

ADead-BIB is not a profiler.

- We don't measure actual execution times
- We estimate based on cost models
- For real profiling, use Nsight, nvprof, or similar tools

Our estimates are based on:
- PCIe 3.0 x16: ~10 GB/s
- GPU compute: ~300 GFLOPS (conservative)
- CPU compute: ~1 GFLOPS (conservative)

---

### ❌ Requires Developer Intent

Some decisions require hints from the developer:

- `will_persist`: Will this data be reused?
- `data_location`: Where is the data now?

We cannot read minds. If the developer doesn't provide intent, we assume worst-case (no persistence, data on host).

---

### ❌ Does NOT Handle All Edge Cases

Current limitations:

- No support for multi-GPU systems
- No CUDA stream awareness
- No async transfer optimization
- No unified memory handling

These are future work, not current features.

---

## What ADead-BIB HEX DOES Do

| Capability | Description |
|------------|-------------|
| **Decision-level governance** | Decides WHEN to use GPU |
| **Misuse detection** | Quantifies bad decisions (0-100) |
| **Contract enforcement** | Guarantees, assumptions, risks |
| **Waste proof** | Proves GPU would be slower |
| **Policy configuration** | YAML-based, auditable |

---

## The Design Philosophy

> **"This system is designed to prevent decision-level waste, not to replace CUDA expertise."**

We believe:

- Developers should understand their workloads
- Systems should prevent obvious mistakes
- Decisions should be auditable
- Complexity should be explicit, not hidden

---

## When NOT to Use ADead-BIB

- If you need kernel-level optimization → Use Nsight, cuBLAS
- If you need adaptive tuning → Use auto-tuning frameworks
- If you need profiling → Use nvprof, Nsight Systems
- If you have multi-GPU workloads → Wait for future versions

---

## Honest Assessment

| Aspect | Status |
|--------|--------|
| Decision-level governance | ✅ Strong |
| Cost model accuracy | 🟡 Approximate |
| Multi-GPU support | ❌ Not implemented |
| Async/stream awareness | ❌ Not implemented |
| Production readiness | 🟡 Prototype-level |

---

> **"The best systems are honest about their limitations."**

*ADead-BIB HEX - Execution Policy Engine*
