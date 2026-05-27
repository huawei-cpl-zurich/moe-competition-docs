# Mixture-of-Experts Documentation

Mixture-of-Experts (MoE) models replace part of a dense neural network with a set of parallel
expert modules and a router. In modern large language models, the MoE component is usually placed
where a dense Transformer would otherwise use a feed-forward network (FFN).

## Dense FFN Versus MoE FFN

In a dense Transformer block, every token passes through the same FFN weights:

```text
token hidden state -> dense FFN -> output
```

In an MoE block, each token is routed to a subset of experts:

```text
token hidden state -> router -> top-k experts -> weighted sum -> output
```

This makes compute sparse. The model may contain many experts, but each token uses only a few of
them.

## Core Terms

**Expert**
: A neural submodule, often an FFN, that processes token hidden states.

**Router or gate**
: A learned function that scores experts for each token and selects the active experts.

**Top-k routing**
: A routing policy that selects the k highest-scoring experts for each token.

**Shared expert**
: An expert that is always active for every token. DeepSeekMoE uses shared experts to carry common
knowledge and reduce redundancy pressure on routed experts.

**Routed expert**
: An expert selected conditionally by the router.

**Expert hotness**
: The observed demand for an expert over a window of tokens or batches. In the simulator, hotness
is the trace-derived load signal used by placement algorithms.

**Expert parallelism**
: A distributed execution strategy where experts are sharded across devices. Tokens or hidden
states are communicated to the devices that host their selected experts.

## Why MoE Models Are Efficient

MoE allows total parameter count and active parameter count to diverge:

- Total parameters determine model capacity.
- Activated parameters determine per-token compute cost.

If a model has 256 routed experts and activates 8 per token, most expert weights are idle for that
token. That sparsity is why MoE can scale model capacity without proportional increases in
per-token FLOPs.

## Why MoE Models Are Hard To Serve

Sparse activation creates irregular load:

- Tokens do not choose experts uniformly.
- Popular domains or token patterns can create hot experts.
- Expert popularity changes over time.
- Distributed MoE serving requires communication between token-owning devices and expert-owning
  devices.

This means the serving system needs to care about both algorithmic routing and hardware placement.

## Load Balancing Mechanisms

Common MoE load-balancing tools include:

- **Auxiliary load-balancing loss** during training.
- **Capacity limits** that cap how many tokens an expert can receive.
- **Noisy routing** to encourage exploration and avoid early expert collapse.
- **Expert grouping** to limit cross-device or cross-node communication.
- **Expert replication** to place extra copies of hot experts.
- **Dynamic placement** to move replicas as hotness changes.

The competition focuses on the last two: replication and dynamic placement.

## Competition Metrics

The simulator reports two main metrics:

**PAR**
: Peak-average ratio of device loads. Lower is better. A PAR near 1 means the most-loaded device is
close to the average device load.

**Transmit amount**
: The amount of expert movement caused by redeployment. Lower is better. A policy that constantly
moves experts may improve balance but create too much deployment cost.

Good submissions should reduce PAR without paying excessive transmit cost.
