# Simulator And Reference Submissions

This page connects the competition API to the reference code in the simulator repository. The
simulator evaluates a participant function named `rebalance` on recent expert-hotness traces and
then measures the resulting load balance and redeployment cost.

## Repository Map

The simulator repository is organized around a small set of entry points:

```text
dynamic_lb_simulator.py       # Original simulator loop and metrics
eplb_algorithms/deepseek.py   # DeepSeek EPLB implementation copied into the repo
experiments/                  # Reproducible sweeps and result tables
submissions/                  # Participant-style reference submissions
trace/                        # Small committed sample traces
```

The full competition traces live on the Codabench worker. The simulator repository includes only
small `LmSys.npy` sample traces for local tests.

## Submission API

Every submission exposes this function:

```python
def rebalance(hotness, n_device, n_red_expert):
    ...
```

The inputs are:

- `hotness`: recent trace window with shape `(collection_window, n_layers, n_experts)`;
- `n_device`: number of expert-parallel devices;
- `n_red_expert`: number of redundant physical expert slots.

The return value is:

```python
(change, layers_priority, deployment_table, aux)
```

`deployment_table` has shape:

```text
(n_layers, n_device, (n_experts + n_red_expert) // n_device)
```

It maps every layer, device, and physical expert slot to a logical expert id.

## Smoke Submission

The smoke submission is the simplest valid API implementation. It builds the default placement and
returns `change=False`, so the simulator keeps the current deployment.

```python
def rebalance(hotness, n_device, n_red_expert):
    n_layers = hotness.shape[1]
    n_experts = hotness.shape[2]
    n_exp_per_dev = (n_experts + n_red_expert) // n_device

    deployment = np.zeros((n_layers, n_device, n_exp_per_dev), dtype=np.int64)

    for layer in range(n_layers):
        for device in range(n_device):
            for slot in range(n_exp_per_dev - 1):
                deployment[layer, device, slot] = (
                    device * (n_exp_per_dev - 1) + slot
                ) % n_experts
            deployment[layer, device, -1] = deployment[layer, device, -2]

    return False, [], deployment, None
```

Walkthrough:

1. Read the model shape from `hotness`.
2. Allocate one deployment table for all layers.
3. Fill each device with a deterministic round-robin logical expert assignment.
4. Duplicate the last base slot into the redundant slot.
5. Return `False` so no redeployment is scheduled.

This is useful for checking packaging and API compatibility, but it is not intended to be
competitive.

## Hot-Expert Baseline Submission

The hot-expert baseline uses the collection window to identify each layer's hottest experts and
places those experts into the redundant slots.

```python
def rebalance(hotness, n_device, n_red_expert):
    load = hotness.sum(axis=0)
    n_layers, n_experts = load.shape
    n_exp_per_dev = (n_experts + n_red_expert) // n_device

    deployment = np.zeros((n_layers, n_device, n_exp_per_dev), dtype=np.int64)
    base_slots = n_exp_per_dev - 1

    for layer in range(n_layers):
        for device in range(n_device):
            for slot in range(base_slots):
                deployment[layer, device, slot] = (
                    device * base_slots + slot
                ) % n_experts

        hottest = np.argsort(load[layer])[::-1]
        for device in range(n_device):
            deployment[layer, device, -1] = hottest[device % len(hottest)]

    layers_priority = np.arange(n_layers, dtype=np.int64)
    return True, layers_priority, deployment, None
```

Walkthrough:

1. Sum the trace window over time to estimate per-layer expert demand.
2. Fill the base slots with a deterministic placement so every logical expert is covered.
3. Sort experts by load in each layer.
4. Use the redundant slot on each device for one of the hottest experts.
5. Request redeployment for every layer in layer order.

This baseline can reduce PAR when hot experts are persistent, but it can also move many slots
because it always returns `change=True`.

## DeepSeek EPLB Walkthrough

DeepSeek EPLB is a placement algorithm for replicated experts. The simulator copy exposes the
entry point:

```python
phy2log, log2phy, logcnt = rebalance_experts(
    weight,
    num_replicas,
    num_groups,
    num_nodes,
    num_gpus,
    enable_hierarchical,
)
```

The key internal stages are:

1. Convert recent token or hotness statistics into per-layer expert weights.
2. Optionally group logical experts and pack those groups across nodes.
3. Replicate hot logical experts by repeatedly assigning extra physical slots to the current
   largest `weight / replica_count`.
4. Pack the resulting physical experts onto GPUs so each GPU receives the same number of experts
   and similar estimated load.
5. Return physical-to-logical and logical-to-physical maps, plus the replica count per logical
   expert.

The core replication step is:

```python
for i in range(num_log, num_phy):
    redundant_indices = (weight / logcnt).max(dim=-1).indices
    phy2log[:, i] = redundant_indices
    rank[:, i] = logcnt[arangen, redundant_indices]
    logcnt[arangen, redundant_indices] += 1
```

The expression `weight / logcnt` estimates the load each replica would carry. Adding the next
replica to the largest value greedily reduces the highest per-replica pressure.

The balanced packing step then sorts objects by weight and repeatedly places the next heaviest
object into the least-loaded pack that still has capacity:

```python
for group in indices[i]:
    pack = min(
        (i for i in range(num_packs) if pack_items[i] < groups_per_pack),
        key=pack_weights.__getitem__,
    )
    pack_index[i, group] = pack
    rank_in_pack[i, group] = pack_items[pack]
    pack_weights[pack] += weight[i, group]
    pack_items[pack] += 1
```

In competition terms, DeepSeek EPLB is the baseline to beat: a strong submission should improve
modeled runtime by lowering PAR enough to justify any additional expert movement.
