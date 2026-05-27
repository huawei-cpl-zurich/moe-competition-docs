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

## Evaluation Loop

At a high level:

1. The simulator initializes an expert deployment.
2. It computes PAR for each iteration from the current deployment and hotness.
3. After a collection window, the algorithm receives recent hotness.
4. The algorithm proposes a new deployment and a layer redeployment order.
5. The simulator charges transmit amount for changed expert slots.
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

The Codabench leaderboard uses `composite_score` as the primary score. Higher is better.

For each evaluated case, the scorer compares the submission against the DS-EPLB baseline for the
same dataset, model, and EP size:

```text
par_ratio = ds_eplb_mean_par / submission_mean_par
transmit_ratio = submission_transmit_amount / ds_eplb_transmit_amount
transmit_adjustment = 25 * (1 - transmit_ratio)

case_score = 100 * par_ratio + transmit_adjustment
```

The final leaderboard score is the arithmetic mean over all evaluated cases:

```text
composite_score = mean(case_score over evaluated cases)
```

This means:

- matching DS-EPLB PAR with no excess transmission gives about 100 points for that case;
- improving PAR over DS-EPLB gives more than 100 points;
- worse PAR gives fewer than 100 points;
- using less transmission than DS-EPLB gives a bonus;
- zero transmission gives a +25 point transmit bonus for that case;
- transmission above DS-EPLB gives a linear penalty.

The raw leaderboard columns are still reported so participants can inspect the tradeoff:

```text
mean_par
transmit_amount
par_vs_ds_eplb
transmit_vs_ds_eplb
```
