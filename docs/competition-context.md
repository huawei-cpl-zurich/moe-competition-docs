# Competition Context

The MoE competition is about dynamic expert placement under expert-parallel inference.

Participants are not asked to train a language model or change DeepSeek's token router. They are
asked to consume expert hotness traces and produce deployment tables that balance load across
devices while avoiding excessive redeployment cost.

## Simulator Inputs

Each trace records expert hotness over iterations:

```text
iteration -> layer -> expert hotness
```

The simulator evaluates multiple models, datasets, and expert-parallel sizes. The original simulator
includes DeepSeek-R1-style and Qwen3-style trace shapes, but the competition bundle keeps traces on
the remote worker rather than in GitHub.

## Placement Output

A placement algorithm returns a deployment table:

```text
layer -> device -> expert slot -> logical expert id
```

If redundant experts are available, the table can contain multiple physical copies of hot logical
experts. This lets load for that expert be divided across devices.

When a policy requests redeployment, only the layers in its priority list are applied. For each
listed layer, the deployment table row is a full replacement for every physical slot in that layer,
not just a list of replicas to add.

## Evaluation Loop

At a high level:

1. The simulator initializes an expert deployment.
2. It computes PAR for each iteration from the current deployment and hotness.
3. After a collection window, the algorithm receives recent hotness.
4. The algorithm proposes a new deployment and a layer redeployment order.
5. The simulator charges transmit amount for changed expert slots, compared slot by slot.
6. The loop repeats as hotness changes.

## What Makes A Good Policy

A strong dynamic placement policy should:

- identify persistent hot experts rather than reacting to every short spike;
- allocate redundant replicas where they reduce peak device load;
- preserve stable placements when the benefit of moving is small;
- consider that moving many expert weights can erase the benefit of better load balance.

## How To Read PAR

PAR is the ratio between the most-loaded device and the average device:

```text
PAR = max(device_loads) / mean(device_loads)
```

Lower is better. A value of 1 is perfectly balanced.

## How To Read Transmit Amount

Transmit amount counts changed expert placements during redeployment. Lower is better. A policy
with zero transmit amount is stable but may leave severe load imbalance; a policy with excessive
movement may improve PAR but be impractical to serve.

## Composite Score

The Codabench leaderboard uses `composite_score` as the primary score. Higher is better. The score
models runtime as worst-device compute time plus expert-transfer overhead.

The modeled runtime is:

```text
modeled_time = compute_time + transfer_time
compute_time = balanced_compute_seconds * mean_par
transfer_time = transmit_amount * expert_bytes / bandwidth_bytes_per_second
```

The fixed hardware assumptions are:

```text
balanced_compute_seconds = 60.0
expert_bytes = 88_080_384
bandwidth_bytes_per_second = 900_000_000_000
```

These correspond to BF16 DeepSeek-style experts with approximately
`3 * 7168 * 2048` parameters per expert and an H100-class high-performance interconnect.

For each evaluated case, the scorer compares modeled runtime against the DS-EPLB baseline for the
same dataset, model, and EP size:

```text
submission_modeled_time =
    60.0 * submission_mean_par
    + submission_transmit_amount * 88_080_384 / 900_000_000_000

baseline_modeled_time =
    60.0 * ds_eplb_mean_par
    + ds_eplb_transmit_amount * 88_080_384 / 900_000_000_000

case_score = 100 * baseline_modeled_time / submission_modeled_time
composite_score = mean(case_score over evaluated cases)
```

This means:

- `100` means equal modeled runtime to DS-EPLB for that case;
- higher than `100` means faster modeled runtime than DS-EPLB;
- lower than `100` means slower modeled runtime than DS-EPLB;
- lower PAR helps by reducing compute time;
- lower transmit helps by reducing transfer time.

The raw leaderboard columns are still reported so participants can inspect the tradeoff:

```text
mean_par
transmit_amount
par_vs_ds_eplb
transmit_vs_ds_eplb
```
