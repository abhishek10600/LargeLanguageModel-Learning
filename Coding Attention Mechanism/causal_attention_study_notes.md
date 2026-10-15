# Causal Self-Attention in PyTorch — Study Notes

Chat Link: https://chatgpt.com/share/6ad0c5cd-addc-83e8-8691-4f509776c7e7

Code Link: https://github.com/abhishek10600/LargeLanguageModel-Learning/blob/main/Coding%20Attention%20Mechanism/causal_attention_mechanism.ipynb

## 1. What this code implements

This code implements a **causal self-attention** layer. It creates Query, Key, and Value vectors from token embeddings, calculates similarity scores between tokens, masks future tokens, applies softmax to obtain attention weights, and combines Value vectors to produce context vectors.

Causal attention is used in autoregressive language models so that a token's representation cannot use information from tokens that come later in the sequence.

## 2. Input tensors and dimensions

The input sentence is:

> Your journey starts with one step.

Each of the six tokens is represented by a vector of three numbers.

```python
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your
    [0.55, 0.87, 0.66],  # journey
    [0.57, 0.85, 0.64],  # starts
    [0.22, 0.58, 0.33],  # with
    [0.77, 0.25, 0.10],  # one
    [0.05, 0.80, 0.55],  # step
])
```

The shape is `(6, 3)`:

- `6` = number of tokens, also called sequence length.
- `3` = number of values in each token embedding, also called input dimension \(d\_{\text{in}}\).

```python
d_in = inputs.shape[1]  # 3
d_out = 2
```

`d_out = 2` means each Query, Key, and Value vector will have two features in this example. It is a small educational dimension; real Transformer models often use much larger dimensions.

### Creating a batch

```python
batch = torch.stack((inputs, inputs), dim=0)
```

This stacks two copies of the same sequence into a batch. Its shape is:

\[
X \in \mathbb{R}^{B \times T \times d\_{\text{in}}}
= \mathbb{R}^{2 \times 6 \times 3}
\]

where:

- \(B=2\): batch size
- \(T=6\): number of tokens
- \(d\_{\text{in}}=3\): input embedding dimension

Both batch examples are identical, so with the same model weights and dropout set to zero, their outputs are identical too.

## 3. The attention layer

```python
class CausalAttention(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        context_length,
        dropout,
        qkv_bias=False
    ):
        super().__init__()

        self.W_query = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_key = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_value = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, input_dimension = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5,
            dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        context_vectors = attention_weights @ values
        return context_vectors
```

## 4. Query, Key, and Value

Each input token embedding is projected into three learned representations:

- **Query (Q):** what this token is looking for.
- **Key (K):** what information this token can offer or how it can be matched.
- **Value (V):** the information that will be mixed into the output if this token receives attention.

These are useful intuitions, not literal descriptions of what the vectors must represent.

Each `nn.Linear(input_dimension, output_dimension)` layer learns a weight matrix and, optionally, a bias. With `qkv_bias=False`, the bias is disabled.

For an input token vector \(x_i\):

\[
q_i=x_iW_Q,\qquad k_i=x_iW_K,\qquad v_i=x_iW_V
\]

PyTorch's `nn.Linear` stores weights in shape `(out_features, in_features)` and internally computes the equivalent of \(x*iW^\top\) if the mathematical weight is written as \(W\in\mathbb{R}^{d*{\text{in}}\times d\_{\text{out}}}\).

For the whole batch:

| Tensor    | Shape       |
| --------- | ----------- |
| `x`       | `(2, 6, 3)` |
| `queries` | `(2, 6, 2)` |
| `keys`    | `(2, 6, 2)` |
| `values`  | `(2, 6, 2)` |

The last dimension changes from 3 to 2 because the linear layers project each embedding into the chosen output dimension.

## 5. Attention scores: how much does each token match another?

```python
attention_scores = queries @ keys.transpose(1, 2)
```

`keys` has shape `(B, T, d_out)`, or `(2, 6, 2)`. The operation

```python
keys.transpose(1, 2)
```

swaps dimensions 1 and 2, producing `(2, 2, 6)`.

The batched matrix multiplication is therefore:

\[
(B,T,d_k)@(B,d_k,T)\longrightarrow(B,T,T)
\]

In this example:

\[
(2,6,2)@(2,2,6)=(2,6,6)
\]

Every score compares one token's Query with another token's Key:

\[
s\_{ij}=q_i^\top k_j
\]

- \(i\) identifies the token doing the attending (the Query row).
- \(j\) identifies the token being considered (the Key column).
- A larger score means a stronger learned match before normalization and masking.

The score matrix has one row per query token and one column per key token.

## 6. Causal mask: block future tokens

```python
self.register_buffer(
    "mask",
    torch.triu(torch.ones(context_length, context_length), diagonal=1)
)
```

`torch.ones(6, 6)` creates a matrix of ones. `torch.triu(..., diagonal=1)` keeps only entries strictly above the main diagonal:

\[
M=
\begin{bmatrix}
0&1&1&1&1&1\\
0&0&1&1&1&1\\
0&0&0&1&1&1\\
0&0&0&0&1&1\\
0&0&0&0&0&1\\
0&0&0&0&0&0
\end{bmatrix}
\]

A `1` means “mask this query-key pair.” The diagonal and lower triangle remain zero, allowing a token to attend to itself and earlier tokens.

```python
attention_scores.masked_fill_(mask.bool(), -torch.inf)
```

`masked_fill_` replaces masked scores with negative infinity. After softmax, a masked position gets probability zero because \(e^{-\infty}=0\).

The mask is registered as a buffer so it is part of the module's state and follows the module across devices, but is not a trainable parameter.

The resulting allowed-attention pattern is:

```text
             Key token
             1  2  3  4  5  6
Query  1     ✓  ×  ×  ×  ×  ×
token  2     ✓  ✓  ×  ×  ×  ×
       3     ✓  ✓  ✓  ×  ×  ×
       4     ✓  ✓  ✓  ✓  ×  ×
       5     ✓  ✓  ✓  ✓  ✓  ×
       6     ✓  ✓  ✓  ✓  ✓  ✓
```

Here ✓ means allowed and × means blocked.

## 7. Scale the scores

The standard scaled dot-product attention formula divides each score by the square root of the Key dimension:

\[
\widetilde{s}\_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}}
\]

In this implementation, `d_k = keys.shape[-1] = 2`, so the scores are divided by \(\sqrt{2}\).

Why scale? Dot products can grow in magnitude as the vector dimension grows. Large values can make softmax very peaked and its gradients small. Scaling helps keep the scores at a more manageable scale.

## 8. Softmax: turn scores into probabilities

**Important correction:** the code in the question uses `dim=1`:

```python
torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=1)
```

For the standard causal self-attention mechanism, this should be `dim=-1` (equivalently, `dim=2`):

```python
attention_weights = torch.softmax(
    attention_scores / (keys.shape[-1] ** 0.5),
    dim=-1
)
```

The score tensor has shape `(B, query_tokens, key_tokens)`. We need to normalize across the **key-token dimension**, so each query's row sums to one across the tokens it is allowed to attend to.

Mathematically:

\[
a*{ij}=
\frac{\exp(\widetilde{s}*{ij})}
{\sum*{m=1}^{T}\exp(\widetilde{s}*{im})}
\]

Masked future positions have \(\widetilde{s}\_{ij}=-\infty\), so their attention weights are zero.

With `dim=1`, PyTorch instead normalizes down the query-token dimension for each key column. That is not the intended attention distribution: rows do not generally sum to one, and the result is not standard causal self-attention. Use `dim=-1`.

## 9. Dropout

```python
self.dropout = nn.Dropout(dropout)
attention_weights = self.dropout(attention_weights)
```

Dropout randomly zeros some elements during training and scales the remaining elements according to the dropout probability. It can regularize the model.

Here:

```python
dropout = 0.0
```

No attention weights are dropped, so dropout has no effect.

## 10. Context vectors: combine Values

```python
context_vectors = attention_weights @ values
```

The attention weights tell us how much each allowed token contributes. The layer uses them to take a weighted sum of the Value vectors:

\[
z*i=\sum*{j=1}^{T}a\_{ij}v_j
\]

In matrix form:

\[
Z=AV
\]

Shapes:

\[
(B,T,T)@(B,T,d_v)\longrightarrow(B,T,d_v)
\]

In this example:

\[
(2,6,6)@(2,6,2)=(2,6,2)
\]

Each output token now contains a context-aware combination of Value vectors from the current and previous tokens. Since the output dimension is 2, the final output shape is `(2, 6, 2)`.

## 11. The complete mathematical mechanism

Let \(X\) be the input sequence, \(W_Q,W_K,W_V\) the learned projection matrices, and \(d_k\) the Key dimension.

### Step 1: Project the input

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

### Step 2: Compute scaled dot-product scores

\[
S=\frac{QK^\top}{\sqrt{d_k}}
\]

For a batch, the transpose is applied to the final two dimensions of \(K\).

### Step 3: Apply the causal mask

Let \(C*{ij}=0\) if \(j\le i\) and \(C*{ij}=-\infty\) if \(j>i\).

\[
S\_{\text{causal}}=S+C
\]

This prevents each position from using future positions.

### Step 4: Normalize across keys

\[
A=\operatorname{softmax}(S\_{\text{causal}})
\]

Softmax is applied independently to each query row, across the final/key dimension.

### Step 5: Combine Value vectors

\[
\boxed{Z=\operatorname{softmax}
\left(\frac{QK^\top}{\sqrt{d_k}}+C\right)V}
\]

With dropout, the attention matrix used for the final multiplication is the dropped-out version of \(A\).

## 12. Initialization and output

```python
torch.manual_seed(789)
context_length = batch.shape[1]
dropout = 0.0

causal_attention = CausalAttention(d_in, d_out, context_length, dropout)
output = causal_attention(batch)
```

- `torch.manual_seed(789)` makes random initialization reproducible for the same relevant PyTorch setup.
- `context_length = 6` makes a \(6\times6\) causal mask.
- `CausalAttention(3, 2, 6, 0.0)` constructs the layer.
- Calling `causal_attention(batch)` invokes its `forward` method.
- `grad_fn=<...>` in the displayed tensor indicates that PyTorch is tracking the operations for automatic differentiation.

Because both batch entries contain the same input, and the same weights are applied to each entry, their output tensors are identical. With dropout set to zero, there is no dropout randomness to distinguish them.

**Note:** the numerical output printed by the original code uses `softmax(..., dim=1)`. If you change this to `dim=-1`, the numerical values will change because you are correcting the attention normalization.

## 13. Shape summary

| Operation                          | Shape       |
| ---------------------------------- | ----------- |
| Input `x`                          | `(2, 6, 3)` |
| Query `Q`                          | `(2, 6, 2)` |
| Key `K`                            | `(2, 6, 2)` |
| Value `V`                          | `(2, 6, 2)` |
| Transposed Key `K.transpose(1, 2)` | `(2, 2, 6)` |
| Attention scores `Q @ Kᵀ`          | `(2, 6, 6)` |
| Causal mask                        | `(6, 6)`    |
| Attention weights `A`              | `(2, 6, 6)` |
| Context vectors `A @ V`            | `(2, 6, 2)` |

## 14. Key takeaways

1. **Query, Key, and Value** are learned projections of the same input in self-attention.
2. **Dot products** measure learned compatibility between each query and each key.
3. **Scaling by \(\sqrt{d_k}\)** controls score magnitude.
4. **The causal mask** prevents access to future tokens.
5. **Softmax must be applied across the key dimension** (`dim=-1`) so each query's allowed attention weights sum to one.
6. **The Value vectors are weighted and summed** to form context-aware token representations.
7. **The correct output shape** in this example is `(batch_size, sequence_length, output_dimension)` = `(2, 6, 2)`.

### Corrected softmax section

The minimal code fix is:

```python
attention_weights = torch.softmax(
    attention_scores / (keys.shape[-1] ** 0.5),
    dim=-1
)
```

This makes the attention weights normalize over key positions, which is the standard causal self-attention behavior.
