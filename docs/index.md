# MoE Competition Documentation

This site gives competition participants the technical background needed to reason about
Mixture-of-Experts (MoE) load balancing, with a focus on DeepSeek-style sparse expert models.

The competition simulator evaluates dynamic expert placement policies using trace-derived expert
hotness. These notes explain why expert routing creates load imbalance, how DeepSeek-style MoE
models route tokens, and what the simulator metrics are intended to capture.

```{toctree}
:maxdepth: 2
:caption: Contents

deepseek-overview
transformers-intro
mixture-of-experts
deepseek-moe-routing
competition-context
references
```
