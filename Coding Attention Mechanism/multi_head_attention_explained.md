# Understanding Multi-Head Attention — Code and Mathematics

These notes explain the Multi-Head Attention notebook step by step, beginning with intuition and then connecting each idea to the mathematics and PyTorch code.

The notebook implements attention in two ways:

1. **From scratch:** calculates attention scores, applies a causal mask, uses softmax, and combines values explicitly.
2. **Using PyTorch:** uses the built-in scaled dot-product attention function, which may use optimized implementations when supported.

---

## 1. The Big Picture

Imagine GPT receives the sentence:

> The cat sat on the mat.

When processing `sat`, the model can use information from other tokens. `cat` helps explain who sat, and `mat` helps explain where the cat sat.

Attention lets the model determine **which other tokens are useful for understanding each token**.

Attention uses three learned representations:

- **Query (Q):** What information am I looking for?
- **Key (K):** What information does each token advertise that it contains?
- **Value (V):** What information should I receive from that token?

These are numerical representations learned during training, not literal questions or labels attached to words.

### Overall attention flow

1. Input embeddings
2. Create queries, keys, and values
3. Calculate attention scores
4. Scale scores
5. Apply the causal mask
6. Apply softmax to get attention weights
7. Combine value vectors
8. Merge the heads and apply an output projection

---

## 2. Understand Tensor Dimensions

A typical input tensor has shape:

```python
(batch_size, num_tokens, embedding_dimension)
```

Each dimension has a distinct meaning:

| Dimension | Meaning | Example |
|---|---|---:|
| `batch_size` | Number of text sequences processed together | 8 |
| `num_tokens` | Number of tokens in each sequence | 4 |
| `embedding_dimension` | Number of numbers representing each token | 256 |

The notebook uses:

```python
output_dim = 256
max_length = 4
batch_size = 8
```

Therefore, the input tensor has shape:

\[
X \in \mathbb{R}^{8 \times 4 \times 256}
\]

In plain English:

- 8 sequences are processed at once.
- Each sequence has 4 tokens.
- Each token is represented by 256 numbers.

A tensor's shape tells you how data is organized, not what its values mean.

---

## 3. Preparing the Input

### 3.1 Import PyTorch

```python
import torch

print(torch.__version__)
```

PyTorch provides tensors, mathematical operations, and neural-network layers.

A tensor can be thought of as:

- Scalar: one number
- Vector: a list of numbers
- Matrix: rows and columns of numbers
- Higher-dimensional tensor: a collection of matrices or more complex structures

Example:

```python
x = torch.tensor([2.0, 3.0, 4.0])
```

Mathematically:

\[
x = [2,3,4]
\]

The `@` operator performs matrix multiplication, and `torch.softmax()` converts scores into normalized weights.

### 3.2 Select the computing device

The notebook checks whether Apple MPS or NVIDIA CUDA is available, otherwise using the CPU:

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

**Note:** selecting a device does not automatically move models and tensors to it. The notebook would also need `.to(device)` calls to run those objects on the selected device.

### 3.3 Tokenize text

The notebook uses:

```python
tokenizer = tiktoken.get_encoding("gpt2")
```

A tokenizer converts text into integer token IDs. For example, a sentence might become a sequence of IDs such as:

```text
[10, 20, 30, 40, 50]
```

These numbers are illustrative. The model processes token IDs rather than raw text, and embeddings later convert IDs into vectors.

### 3.4 Create input and target sequences

The dataset code slices token IDs:

```python
input_chunk = token_ids[i:i + max_length]
target_chunk = token_ids[i + 1: i + max_length + 1]
```

Suppose the token IDs are:

```text
[10, 20, 30, 40, 50]
```

With `max_length = 4`:

| Tensor | Token IDs |
|---|---|
| `input_chunk` | 10, 20, 30, 40 |
| `target_chunk` | 20, 30, 40, 50 |

This shift trains a language model to predict the next token:

- Given `10`, predict `20`.
- Given `10, 20`, predict `30`.
- Given `10, 20, 30`, predict `40`.

The attention layer helps create the representations used for these predictions. The notebook does not implement the complete GPT model or its training loop.

### 3.5 Create token and positional embeddings

```python
token_embeddings = token_embedding_layer(x)

pos_embeddings = pos_embedding_layer(
    torch.arange(max_length)
)

input_embeddings = token_embeddings + pos_embeddings
```

Token embeddings represent tokens. Positional embeddings provide information about their positions.

The token embedding layer is created with:

```python
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
```

With a vocabulary size of 50,257 and an embedding dimension of 256, its embedding matrix has shape:

\[
E \in \mathbb{R}^{50257 \times 256}
\]

If a token has ID 100, PyTorch retrieves row 100 from the embedding matrix. It does not multiply the ID by a number.

The positional embedding layer is:

```python
pos_embedding_layer = nn.Embedding(context_length, output_dim)
```

The final input representation is the sum:

\[
X = E_{\text{token}} + E_{\text{position}}
\]

For example:

\[
[0.2,0.5,0.1]+[0.1,0.0,0.3]=[0.3,0.5,0.4]
\]

The actual vectors have 256 components, but the addition works the same way.

### 3.6 Check the shape

```python
print(input_embeddings.shape)
```

For 8 sequences, 4 tokens per sequence, and 256 dimensions per token:

```text
torch.Size([8, 4, 256])
```

This is the input to Multi-Head Attention.

---

## 4. The Core Attention Equation

The central equation is:

\[
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
}
\]

It consists of four operations:

1. \(QK^\top\): calculate similarity scores.
2. Divide by \(\sqrt{d_k}\): scale the scores.
3. Softmax: convert scores into weights.
4. Multiply by \(V\): combine information.

### 4.1 Query, Key, and Value

The notebook creates the three tensors with learned linear layers:

```python
keys = self.W_key(x)
queries = self.W_query(x)
values = self.W_value(x)
```

Mathematically:

\[
Q = XW_Q
\]

\[
K = XW_K
\]

\[
V = XW_V
\]

Where:

- \(X\): input embeddings
- \(W_Q\): query weight matrix
- \(W_K\): key weight matrix
- \(W_V\): value weight matrix
- \(Q,K,V\): resulting tensors

These weight matrices are learned during training. A forward pass alone does not train them.

#### A tiny example

Let:

\[
X =
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

Suppose, just for this simplified example:

\[
W_Q =
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
W_K =
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

Then:

\[
Q=XW_Q=X,\qquad K=XW_K=X
\]

The query and key vectors happen to equal the input vectors in this example. In a real model, the learned matrices generally transform them into different representations.

The three transformations serve different purposes:

- Query: what a token is looking for
- Key: how a token can be matched to a query
- Value: what information the token contributes

A token's query is compared with other tokens' keys. The resulting attention weights determine how their values are combined.

### 4.2 Calculate attention scores

Code:

```python
attn_scores = queries @ keys.transpose(2, 3)
```

Mathematically:

\[
S=QK^\top
\]

For a tiny example:

\[
Q =
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
K =
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

Then:

\[
S =
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}^{\top}
=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

Rows represent tokens asking for information; columns represent tokens being compared against them.

In this example:

- Token 1 scores itself as 1 and token 2 as 0.
- Token 2 scores token 1 as 0 and itself as 1.

These are raw scores, not probabilities. They may be positive, negative, or zero.

For a sequence of \(T\) tokens, the score matrix for one head has shape:

\[
S \in \mathbb{R}^{T \times T}
\]

Every query position is compared with every key position before masking.

### 4.3 Scale the scores

Code:

```python
attn_scores / keys.shape[-1]**0.5
```

Mathematically:

\[
S_{\text{scaled}}=\frac{QK^\top}{\sqrt{d_k}}
\]

Here, \(d_k\) is the dimension of a key vector in one head.

The notebook uses an output dimension of 256 and two heads:

\[
d_k=\frac{256}{2}=128
\]

Therefore:

\[
\sqrt{128}\approx 11.31
\]

Why scale?

As the number of dimensions increases, dot products can become large in magnitude. Large scores can make softmax excessively concentrated, resulting in very small gradients for many inputs. Dividing by \(\sqrt{d_k}\) helps control the score magnitudes.

For example, if a raw score is 8 and \(d_k=4\):

\[
\frac{8}{\sqrt{4}}=\frac{8}{2}=4
\]

Scaling does not change which score is largest when all scores are divided by the same positive constant. It changes how sharply softmax distinguishes between scores.

### 4.4 Softmax: convert scores into weights

Code:

```python
attn_weights = torch.softmax(
    attn_scores / keys.shape[-1]**0.5,
    dim=-1
)
```

The softmax equation is:

\[
\operatorname{softmax}(s_i)
=
\frac{e^{s_i}}{\sum_j e^{s_j}}
\]

Example: scores \([2,1,0]\).

First calculate exponentials:

\[
[e^2,e^1,e^0]\approx[7.389,2.718,1]
\]

Their sum is:

\[
7.389+2.718+1=11.107
\]

Divide each exponential by the total:

\[
[0.665,0.245,0.090]
\]

Their sum is approximately 1:

\[
0.665+0.245+0.090=1
\]

The model assigns about 66.5% of the attention weight to the first token, 24.5% to the second, and 9% to the third.

These are **attention weights**, not the model's final probabilities for the next word.

The argument `dim=-1` means softmax operates over the last dimension, which contains the key positions being attended to.

---

## 5. Causal Masking: Prevent Looking into the Future

Suppose the input is:

> The cat sat down.

When predicting the next token after `The`, GPT must not use information from `cat`, `sat`, or `down`. Those tokens occur later in the sequence.

The notebook creates a causal mask:

```python
self.register_buffer(
    "mask",
    torch.triu(
        torch.ones(context_length, context_length),
        diagonal=1
    )
)
```

For four tokens, the mask is:

\[
M =
\begin{bmatrix}
0&1&1&1\\
0&0&1&1\\
0&0&0&1\\
0&0&0&0
\end{bmatrix}
\]

- `0` means attention is allowed.
- `1` means attention is forbidden.

The first token can attend only to itself. The second token can attend to the first token and itself. The third can attend to the first two tokens and itself.

The implementation applies the mask:

```python
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

attn_scores.masked_fill_(mask_bool, -torch.inf)
```

Forbidden scores become negative infinity:

\[
-\infty
\]

Because:

\[
e^{-\infty}=0
\]

softmax assigns exactly zero attention weight to future tokens.

This is **causal self-attention**. The diagonal is not masked, so each token can attend to itself.

---

## 6. Combine Information Using Values

After calculating attention weights, the model uses them to combine the value vectors.

Code:

```python
context_vec = attn_weights @ values
```

Mathematically:

\[
C=AV
\]

Where:

- \(A\): attention-weight matrix
- \(V\): value matrix
- \(C\): contextual representation

### Numerical example

Suppose three tokens have one-dimensional value vectors:

\[
V =
\begin{bmatrix}
10\\
20\\
30
\end{bmatrix}
\]

Suppose the third token has attention weights:

\[
A_3=[0.2,0.3,0.5]
\]

Its contextual representation is:

\[
C_3=0.2(10)+0.3(20)+0.5(30)
\]

\[
C_3=2+6+15
\]

\[
\boxed{C_3=23}
\]

If the weights were instead \([0.8,0.1,0.1]\):

\[
C_3=0.8(10)+0.1(20)+0.1(30)=13
\]

Different attention weights produce different contextual representations.

In the actual model, each value is a vector of 128 numbers per head, not one number. The weighted sum is performed across every vector dimension.

**Remember:** values contain the information being combined; attention weights determine how much each token contributes.

---

## 7. What Makes It Multi-Head Attention?

A single attention head learns one set of query, key, and value transformations. Multi-head attention runs multiple heads in parallel so that the model can learn different patterns.

For example, one head might learn to focus on nearby words, while another might learn to connect a pronoun with a relevant noun. These are possibilities, not guaranteed roles for particular heads.

The notebook creates two heads:

```python
mha = MultiHeadAttention(
    d_in,
    d_out,
    context_length,
    0.0,
    num_heads=2
)
```

The output dimension is 256 and there are two heads.

The dimension of each head is:

\[
d_{\text{head}}=\frac{d_{\text{out}}}{h}
\]

Where \(h\) is the number of heads.

For this notebook:

\[
d_{\text{head}}=\frac{256}{2}=128
\]

The model projects each token into 256 query features, 256 key features, and 256 value features, then splits each 256-dimensional vector into two 128-dimensional heads.

The two head outputs are concatenated:

\[
128+128=256
\]

The model can then combine information from both heads while preserving the overall embedding dimension.

More heads do not automatically mean better results. The number of heads and their dimensions are architectural choices.

---

## 8. The `MultiHeadAttention` Class, Step by Step

### Step A: Initialize the layer

```python
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False
    ):
        super().__init__()
        assert output_dimension % num_heads == 0
```

`nn.Module` is PyTorch's base class for neural-network modules.

The assertion ensures the output dimension can be divided evenly among the heads. With 256 dimensions and two heads, each gets 128.

### Step B: Calculate the head dimension

```python
self.output_dimension = output_dimension
self.num_heads = num_heads

self.head_dim = output_dimension // num_heads
```

The `//` operator performs integer division:

\[
256 // 2=128
\]

### Step C: Create learned projections

```python
self.W_query = nn.Linear(
    input_dimension, output_dimension, bias=qkv_bias
)

self.W_key = nn.Linear(
    input_dimension, output_dimension, bias=qkv_bias
)

self.W_value = nn.Linear(
    input_dimension, output_dimension, bias=qkv_bias
)
```

A linear layer performs:

\[
y=xW^\top+b
\]

PyTorch stores weights in the orientation used by `nn.Linear`, so the conventional formula includes a transpose.

With 256 input and output features, each weight matrix has shape \(256\times256\) in conventional mathematical notation. Bias is disabled by default here.

These are three separate learned transformations.

### Step D: Create the output projection

```python
self.out_proj = nn.Linear(input_dimension, output_dimension)
```

After the heads are combined, this layer mixes their concatenated features:

\[
Y=CW_O^\top+b_O
\]

Here, the input and output dimensions are both 256.

### Step E: Dropout

```python
self.dropout = nn.Dropout(dropout)
```

Dropout randomly zeros some elements during training to help reduce overfitting. The notebook passes `dropout=0.0`, so dropout has no practical effect in this example.

### Step F: Build the causal mask

```python
self.register_buffer(
    "mask",
    torch.triu(
        torch.ones(context_length, context_length),
        diagonal=1
    )
)
```

`torch.triu` retains the upper triangular portion of a matrix. With `diagonal=1`, the main diagonal is excluded, leaving future positions marked.

`register_buffer` stores the mask with the module without making it a trainable parameter. The buffer also follows the module when it is moved between devices.

### Step G: Read the input shape

```python
b, num_tokens, input_dimension = x.shape
```

For an input of shape `(8, 4, 256)`:

- `b = 8`
- `num_tokens = 4`
- `input_dimension = 256`

The projections produce:

```python
keys = self.W_key(x)
queries = self.W_query(x)
values = self.W_value(x)
```

Each tensor has shape:

```text
(8, 4, 256)
```

### Step H: Split each tensor into heads

```python
keys = keys.view(
    b, num_tokens, self.num_heads, self.head_dim
)
```

The same operation is applied to queries and values.

The shape changes from:

```text
(8, 4, 256)
```

to:

```text
(8, 4, 2, 128)
```

No values are lost or mathematically calculated in this step. The values are reorganized into two groups of 128 for each token.

### Step I: Transpose the dimensions

```python
keys = keys.transpose(1, 2)
queries = queries.transpose(1, 2)
values = values.transpose(1, 2)
```

The shape becomes:

```text
(8, 2, 4, 128)
```

This places the head dimension before the token dimension so attention can be calculated independently for each head.

### Step J: Calculate scores

```python
attn_scores = queries @ keys.transpose(2, 3)
```

The query tensor has shape:

\[
(8,2,4,128)
\]

The transposed key tensor has shape:

\[
(8,2,128,4)
\]

The inner dimensions match, producing:

\[
\boxed{(8,2,4,4)}
\]

This means eight sequences, two heads, four query positions, and four key positions.

### Step K: Mask, scale, and normalize

```python
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

attn_scores.masked_fill_(mask_bool, -torch.inf)

attn_weights = torch.softmax(
    attn_scores / keys.shape[-1]**0.5,
    dim=-1
)
```

The code:

1. Replaces future-token scores with \(-\infty\).
2. Divides scores by \(\sqrt{128}\).
3. Applies softmax across the four key positions.

The weights have shape:

```text
(8, 2, 4, 4)
```

Each row sums to 1, apart from floating-point rounding. Masked positions have weight zero.

### Step L: Combine values and merge heads

```python
context_vec = (attn_weights @ values).transpose(1, 2)
```

The matrix multiplication produces:

```text
(8, 2, 4, 128)
```

After transposing:

```text
(8, 4, 2, 128)
```

Then:

```python
context_vec = context_vec.contiguous().view(
    b, num_tokens, self.output_dimension
)
```

The final two dimensions are merged:

\[
2\times128=256
\]

The shape becomes:

```text
(8, 4, 256)
```

`contiguous()` ensures the tensor's memory layout is suitable for the subsequent `view()` operation.

Finally:

```python
context_vec = self.out_proj(context_vec)
return context_vec
```

The output projection mixes information from the heads. The output shape remains:

```text
torch.Size([8, 4, 256])
```

That completes the forward pass of the from-scratch implementation.

---

## 9. The PyTorch Built-in Implementation

The notebook implements attention a second time using PyTorch's built-in scaled dot-product attention.

The key difference is that PyTorch performs the attention calculation internally.

### Step A: Combine Q, K, and V into one linear layer

```python
self.qkv = nn.Linear(
    input_dimension,
    3 * output_dimension,
    bias=qkv_bias
)
```

Previously, there were three separate linear layers. Now one layer produces:

\[
3\times256=768
\]

features, giving an output shape of:

```text
(8, 4, 768)
```

Combining the projections can reduce overhead and improve computational efficiency.

### Step B: Reshape and separate the projections

```python
qkv = qkv.view(
    batch_size,
    num_tokens,
    3,
    self.num_heads,
    self.head_dim
)
```

The shape becomes:

```text
(8, 4, 3, 2, 128)
```

The dimension of size `3` represents Q, K, and V.

Next:

```python
qkv = qkv.permute(2, 0, 3, 1, 4)

queries, keys, values = qkv
```

After `permute`, the shape is:

```text
(3, 8, 2, 4, 128)
```

Separating along the first dimension gives each of Q, K, and V the shape:

```text
(8, 2, 4, 128)
```

This is the same layout used by the from-scratch implementation before calculating attention.

### Step C: Let PyTorch perform attention

```python
context_vec = nn.functional.scaled_dot_product_attention(
    queries,
    keys,
    values,
    attn_mask=None,
    dropout_p=use_dropout,
    is_causal=True
)
```

The function internally performs scaled dot-product attention:

\[
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V
\]

Here, \(M\) represents causal masking. Setting `is_causal=True` prevents each position from attending to future positions.

You do not need to calculate the scores, scale them, apply the mask, or call softmax manually.

#### What about FlashAttention?

FlashAttention is an optimized way to compute exact attention. It can reduce memory traffic by processing attention in tiles instead of materializing the entire attention matrix in the usual way.

PyTorch may select an appropriate optimized backend depending on the device, tensor properties, and software support.

Using `scaled_dot_product_attention` makes optimized implementations available, but it does **not** guarantee that FlashAttention specifically is being used on every system.

### Step D: Merge the heads and project the output

```python
context_vec = (
    context_vec.transpose(1, 2)
    .contiguous()
    .view(
        batch_size,
        num_tokens,
        self.output_dimension
    )
)

context_vec = self.proj(context_vec)
```

These operations:

1. Move the token dimension before the head dimension.
2. Combine the heads into 256 features.
3. Apply an output projection.

The final shape remains:

```text
torch.Size([8, 4, 256])
```

Both implementations aim to perform the same mathematical operation. Because their learned weights are initialized separately, their numerical outputs are not automatically identical.

---

## 10. Compare the Two Implementations

| Operation | From scratch | PyTorch version |
|---|---|---|
| Create Q, K, V | Three linear layers | One combined linear layer |
| Split heads | `view()` | `view()` |
| Rearrange dimensions | `transpose()` | `permute()` |
| Calculate scores | `queries @ keys.T` | Built-in function |
| Scale scores | Explicit division | Built-in function |
| Apply causal mask | Explicit triangular mask | `is_causal=True` |
| Apply softmax | Explicit `torch.softmax()` | Built-in function |
| Combine values | Explicit matrix multiplication | Built-in function |
| Merge heads | `transpose()` and `view()` | `transpose()` and `view()` |

The from-scratch version is useful for learning because every mathematical operation is visible. The built-in version is more convenient for practical implementations.

The first version applies its manually created causal mask. The second enables causal attention using `is_causal=True`. With dropout set to zero and matching projection weights, both implement the same attention calculation, subject to floating-point differences.

---

## 11. Memorize the Shape Transformations

For this example:

| Stage | Shape |
|---|---|
| Input embeddings | `(8, 4, 256)` |
| Q, K, V projections | `(8, 4, 256)` each |
| Split into heads | `(8, 4, 2, 128)` |
| Transpose heads | `(8, 2, 4, 128)` |
| Attention scores | `(8, 2, 4, 4)` |
| Attention weights | `(8, 2, 4, 4)` |
| Weighted values | `(8, 2, 4, 128)` |
| Merge heads | `(8, 4, 256)` |
| Final output | `(8, 4, 256)` |

### Follow one token representation through the layer

1. **256 input features:** each token starts with 256 numbers.
2. **128 features per head:** each of two heads uses 128 dimensions for Q, K, and V.
3. **4 × 4 attention matrix per head:** each of four tokens compares against four token positions.
4. **256 combined output features:** two 128-dimensional head outputs are concatenated and projected.

Remember:

\[
128+128=256
\]

The attention-score matrix has a different shape from the token representations because it describes token-to-token relationships rather than embedding features.

---

## 12. Self-Test

Try answering these questions before checking the answers.

### Question 1: Why calculate Q, K, and V?

A. To convert all numbers into probabilities  
B. To learn how tokens request, match, and contribute information  
C. To remove the embedding dimension  

**Answer: B.** Queries are compared against keys to calculate attention weights. Those weights determine how values are combined.

### Question 2: If the output dimension is 256 and there are 4 heads, what is the dimension of each head?

A. 64  
B. 128  
C. 252  

**Answer: A.** The head dimension is:

\[
256/4=64
\]

### Question 3: Why replace future-token scores with negative infinity?

A. To make the embeddings larger  
B. To make future tokens have zero attention weight after softmax  
C. To remove the current token  

**Answer: B.** Softmax assigns zero weight to negative infinity, preventing a token from using future information.

### Question 4: What does `attn_weights @ values` calculate?

A. A weighted combination of value vectors  
B. The vocabulary size  
C. The number of attention heads  

**Answer: A.** Attention weights determine how strongly each value vector contributes to each token's output.

---

## 13. Final Summary

The essential mathematical story is:

1. **Embeddings** turn token IDs into vectors.
2. **Linear projections** transform the vectors into queries, keys, and values.
3. **Dot products** compare queries against keys.
4. **Scaling** controls the magnitude of attention scores.
5. **Causal masking** prevents future information from leaking into earlier positions.
6. **Softmax** converts scores into attention weights.
7. **Weighted value sums** produce contextual representations.
8. **Multiple heads** learn separate attention patterns, and their outputs are merged.

The central equation to remember is:

\[
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V
}
\]

For the from-scratch implementation, \(M\) represents the causal mask, with forbidden positions set to negative infinity.

**Attention is not the entire GPT model.** GPT also uses residual connections, layer normalization, feed-forward networks, token prediction layers, and training objectives. This notebook focuses on the attention mechanism itself.

### Recommended study approach

Master the small numerical example first:

1. Calculate attention scores.
2. Apply the causal mask.
3. Normalize scores with softmax.
4. Combine values using the weights.
5. Trace the tensor shapes through the two heads.

Once the small example feels intuitive, the larger PyTorch tensors are simply performing the same mathematics in parallel.
