# Introduction To Transformers

Transformers are the neural-network architecture behind modern large language models. They process
sequences of tokens by repeatedly applying attention and feed-forward transformations to a hidden
state for each token.

## Tokens And Hidden States

A text prompt is first split into tokens. Each token is mapped to a vector called an embedding.
After adding positional information, the model updates these vectors through a stack of Transformer
layers.

At layer \(l\), each token has a hidden state:

```text
h_l[token] -> Transformer layer -> h_{l+1}[token]
```

The final hidden states are used to predict the next token.

## Self-Attention

Self-attention lets each token read information from other tokens in the same sequence. A token
produces three vectors:

- **query**: what this token is looking for;
- **key**: what this token offers to other tokens;
- **value**: the information this token contributes.

Attention compares queries with keys, converts those scores into weights, and uses the weights to
mix value vectors. This gives the model a content-dependent way to move information across the
sequence.

## Multi-Head Attention

Transformers usually use multiple attention heads in parallel. Each head can learn a different
pattern, such as local syntax, long-range references, or delimiter matching. The head outputs are
combined and projected back into the model hidden dimension.

DeepSeek-V2 and DeepSeek-V3 use **Multi-head Latent Attention (MLA)**, a variant designed to reduce
key-value cache size during inference.

## Feed-Forward Networks

After attention, a standard Transformer block applies a feed-forward network (FFN) independently to
each token:

```text
attention output -> FFN -> layer output
```

In dense models, every token uses the same FFN parameters. In MoE models, this FFN is replaced by a
router plus multiple expert FFNs, so different tokens can activate different experts.

## Residual Connections And Normalization

Transformer layers use residual connections so each sublayer learns an update rather than a full
replacement:

```text
h = h + attention(norm(h))
h = h + feed_forward(norm(h))
```

Normalization stabilizes training and keeps hidden-state scales manageable.

## Autoregressive Inference

Decoder-only language models generate text one token at a time. At each step:

1. the current token sequence is processed by the Transformer;
2. the model predicts a probability distribution over the next token;
3. a token is selected;
4. the new token is appended and the process repeats.

Serving efficiency depends heavily on attention cache management and FFN compute. MoE models reduce
FFN compute per token, but they introduce routing and expert-placement challenges.

## Why Transformers Lead To MoE

In large dense Transformers, FFN layers contain a large fraction of the parameters and compute. MoE
replaces dense FFNs with sparse expert FFNs. This increases total model capacity while keeping only
a small subset of expert parameters active for each token.

That is the architectural step from dense Transformers to DeepSeek-style MoE models.
