# DeepSeek MoE Routing Algorithm

This page breaks down the DeepSeek-style MoE router at the level needed for load-balancing
experiments. The exact training implementation is not fully reproduced in the public inference
code, but the papers and inference implementation expose the main architecture and inference-time
routing path.

## Router Inputs And Outputs

For each token hidden state \(h_t\), the router produces:

- a score for every routed expert;
- a selected set of routed experts;
- routing weights used to combine selected expert outputs.

The MoE layer also includes shared experts. Shared experts do not need routing; they process every
token.

## DeepSeekMoE Design

DeepSeekMoE differs from a conventional top-k MoE in two important ways.

### Fine-Grained Experts

Instead of keeping a small number of large experts, DeepSeekMoE segments experts more finely. If a
conventional MoE would activate \(K\) experts from \(N\), DeepSeekMoE can activate a larger number
from a larger expert pool while keeping the compute budget controlled by reducing expert size.

The intended effect is more flexible composition: a token can combine several specialized smaller
experts instead of relying on fewer coarse experts.

### Shared Experts

DeepSeekMoE isolates some experts as shared experts. These are always active and are meant to
capture common knowledge. Routed experts can then focus more on specialized information because
they do not need to redundantly learn the common component for every token.

## DeepSeek-V3 Routing Path

The public DeepSeek-V3 inference code implements the gate as a learned linear scoring layer over
routed experts. The 671B configuration uses sigmoid scores, 256 routed experts, 8 active routed
experts per token, 8 expert groups, and 4 selected groups.

At inference time, the route is:

1. Compute expert affinities from the token hidden state and the gate weights.
2. Apply the configured score function. In V3, the large model uses sigmoid scores.
3. Add expert bias for top-k selection when using the auxiliary-loss-free routing method.
4. Partition experts into groups and compute group scores.
5. Keep only the configured number of high-scoring groups.
6. Mask experts outside the selected groups.
7. Select the top-k routed experts from the remaining candidates.
8. Normalize the selected routing weights.
9. Multiply by the route scaling factor.
10. Execute selected routed experts and add shared expert output.

The expert bias in step 3 affects expert selection. The final routing weights used to combine
expert outputs are still based on the original affinity scores, not the biased scores.

## Pseudocode

```text
input: token hidden state h
parameters:
  W_gate                    # one vector per routed expert
  expert_bias               # used for no-aux-load-balancing selection
  n_groups
  topk_groups
  topk_experts
  route_scale

scores = sigmoid(h @ W_gate.T)

selection_scores = scores + expert_bias
group_scores = score_groups(selection_scores, n_groups)
selected_groups = topk(group_scores, topk_groups)

masked_scores = mask_experts_outside_groups(selection_scores, selected_groups)
expert_ids = topk(masked_scores, topk_experts)

weights = gather(scores, expert_ids)
weights = normalize(weights)
weights = route_scale * weights

output = shared_experts(h)
for expert_id, weight in zip(expert_ids, weights):
    output += weight * routed_expert[expert_id](h)
```

## Group-Limited Routing

Group-limited routing reduces the candidate expert set before top-k expert selection. Experts are
divided into groups. The router first chooses a limited number of groups, then selects experts only
inside those groups.

This is important for distributed systems because expert groups can be aligned with physical
topology. A router that can restrict candidates to fewer groups can reduce communication pressure,
although the final load still depends on token distribution and expert popularity.

## Auxiliary-Loss-Free Load Balancing

Traditional MoE systems often add an auxiliary loss to encourage balanced expert usage. DeepSeek-V3
instead describes an auxiliary-loss-free balancing strategy based on expert bias terms.

The bias terms are updated to influence routing decisions toward underused experts and away from
overused experts. The important distinction is:

- biased scores guide top-k selection;
- original affinity scores determine the mixture weights.

This separation aims to improve load balance while reducing the training-quality tradeoff that can
come from forcing balance directly through an auxiliary objective.

## Relation To The Competition Simulator

The simulator does not run token-level routing. Instead, it consumes traces that summarize expert
hotness over time. A hot expert is one that the router selected frequently or heavily in a time
window.

Given these hotness traces, a placement policy must decide:

- which experts deserve redundant replicas;
- where replicas should be placed across devices;
- how quickly to redeploy when hotness changes;
- how to trade lower PAR against higher transmit amount.

DeepSeek routing explains why those traces are non-uniform. The competition asks participants to
solve the downstream systems problem created by that non-uniform routing.
