# DeepSeek Overview

DeepSeek is a family of open-weight language models whose recent large models combine
Transformer blocks with efficiency-oriented architectural choices. For MoE competitions, the
important line is DeepSeekMoE, then the DeepSeek-V2 and DeepSeek-V3 models that use the
DeepSeekMoE design inside larger production-scale systems.

## Why DeepSeek Matters For MoE Load Balancing

DeepSeek-V2 and DeepSeek-V3 are sparse MoE models: each token activates only a subset of the
available feed-forward experts. This allows the model to have many total parameters while keeping
per-token computation much smaller than a dense model of the same total size.

That efficiency comes with a systems problem. The router may send many tokens to the same experts,
which creates:

- uneven compute load across devices;
- uneven all-to-all communication volume;
- hot experts that become placement bottlenecks;
- a need for load balancing during training and careful expert placement during inference.

The simulator used in this competition focuses on the inference placement side: given traces of
expert hotness, an algorithm chooses where redundant expert replicas should live.

## Architecture Lineage

### DeepSeekMoE

The DeepSeekMoE paper introduces two core ideas:

- **Fine-grained expert segmentation**: instead of a smaller number of coarse experts, split experts
  into more smaller experts and activate more of them. This gives the router a richer combination
  space while keeping total activated capacity controlled.
- **Shared expert isolation**: reserve some experts that are always active to capture common
  knowledge, while routed experts specialize in more specific patterns.

The paper positions these mechanisms as a way to encourage expert specialization and reduce
knowledge redundancy among routed experts.

### DeepSeek-V2

DeepSeek-V2 combines two major efficiency techniques:

- **Multi-head Latent Attention (MLA)**, which compresses key-value cache state for efficient
  long-context inference.
- **DeepSeekMoE**, which replaces many dense FFN blocks with sparse expert layers.

For MoE routing, DeepSeek-V2 uses a top-k router over routed experts and shared experts that are
always active.

### DeepSeek-V3

DeepSeek-V3 keeps the MLA + DeepSeekMoE direction and scales it up. Its technical report describes
a 671B-parameter MoE model with 37B parameters activated per token. The official inference
configuration for the 671B model includes:

- 61 Transformer layers;
- 3 early dense layers before MoE layers;
- 256 routed experts;
- 1 shared expert;
- 8 activated routed experts per token;
- 8 expert groups with 4 limited groups used during routing;
- sigmoid routing scores and a routing scale of 2.5.

V3 also introduces an auxiliary-loss-free load-balancing strategy: rather than relying only on an
auxiliary balancing loss added to the training objective, it uses expert-specific bias terms for
routing decisions while keeping the final expert weights derived from the original affinity scores.

## Architecture Components

At a high level, a DeepSeek-V3-style block contains:

1. **Attention path** using MLA.
2. **Feed-forward path**, which is dense in early layers and MoE in later layers.
3. **Router/gate** that scores routed experts for each token.
4. **Shared experts** that process every token.
5. **Routed experts** selected sparsely per token.
6. **Weighted combine** of the selected expert outputs.

The routing path is the piece that creates competition-relevant traces: if the same experts are
selected disproportionately often, the physical devices holding those experts receive more work.

## Practical Interpretation

For this competition, it is useful to separate three layers of the system:

- **Model architecture**: how the router chooses experts for tokens.
- **Runtime routing load**: how often each expert is selected over time.
- **Physical placement**: where each expert or replica is deployed across devices.

The competition does not retrain the router. It asks participants to improve the physical placement
and replica allocation policy given observed hotness traces.
