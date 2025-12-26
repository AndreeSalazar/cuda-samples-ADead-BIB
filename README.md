# ADead-BIB HEX — CUDA Execution Policy Engine

> **Preventing GPU misuse at decision level.**

> *"Above CUDA, below frameworks, next to the runtime."*

---

## TL;DR

- Same CUDA kernels
- Same hardware
- Fewer transfers
- **2× faster pipelines**
- Deterministic decisions

---

## Run the Demo (30 seconds)

```bash
cd ADead_HEX_Portable
cargo run --example pipeline_demo
```

Expected output:

```
CUDA Naive:     2,443 µs, 10 transfers
ADead-BIB HEX:  1,222 µs,  2 transfers

🔥 EFFICIENCY GAIN: 2.0x faster
🔥 TRANSFER REDUCTION: 80%
```

---

## Why This Matters

Most GPU slowdowns are caused by **bad execution decisions**, not bad kernels.

```
Same GPU.
Same kernel.
Different decisions.
10× difference.
```

This project introduces a **policy engine** that lives:

> **Above CUDA, below frameworks, next to the runtime.**

---

## What This Is

An **execution policy engine** that:

- ✅ Decides WHEN to use GPU
- ✅ Detects misuse patterns (Score 0-100)
- ✅ Enforces decision contracts
- ✅ Proves GPU waste before it happens

## What This Is NOT

- ❌ Not a CUDA replacement
- ❌ Not a kernel optimizer
- ❌ Not a framework
- ❌ Not magic

---

## Where to Look Next

| Document | Purpose |
|----------|---------|
| [NVIDIA_MANIFESTO.md](NVIDIA_MANIFESTO.md) | Full pitch for NVIDIA |
| [ONE_MINUTE_DEMO.md](ONE_MINUTE_DEMO.md) | 1-minute narrative demo |
| [CUDA_FAILURE_CASE.md](CUDA_FAILURE_CASE.md) | Real failure cases |
| [ADead_HEX_Portable/](ADead_HEX_Portable/) | Runnable code |

---

## The 3 Key Answers

**"Where does this live?"**
> Above CUDA, below frameworks, next to the runtime.

**"What is this?"**
> An execution policy engine that prevents GPU misuse.

**"Why should NVIDIA care?"**
> Because most GPU slowdowns are decision bugs, not hardware bugs.

---

## Quick Links

```bash
# Full demo with Decision Contracts
cargo run --example full_demo

# Pipeline comparison (2x faster)
cargo run --example pipeline_demo

# Adversarial worst-case demo
cargo run --example adversarial_demo
```

---

*ADead-BIB HEX - Execution Policy Engine*
*"CUDA gives power. ADead-BIB gives judgment."*
