# Transformer Workflow (Detailed Notes)

## 0. Overview
A Transformer processes a token sequence by:

1. converting tokens to vectors (embeddings),
2. mixing information across positions using attention,
3. refining each position with a feed-forward network (FFN),  
repeated for multiple layers.

Main architectures:

- Encoder-only: representation/understanding
- Decoder-only: autoregressive generation
- Encoder–Decoder: conditional generation (seq2seq)

---

## 1. Encoder and Decoder 
### 1.1 Components
- **Encoder**: reads the source sequence and outputs a contextual representation for every source position.
- **Decoder**: generates the target sequence left-to-right.
  - uses **masked self-attention** so it cannot see future target tokens
  - uses **cross-attention** to attend to encoder outputs (memory) 

### 1.2 Canonical workflow (seq2seq)
1. Source sentence → token IDs
2. Source IDs → token embeddings + positional encodings → source matrix
3. Source matrix → encoder stack → memory
4. Target prefix (starts with a begin token) → token embeddings + positional encodings
5. Target prefix → decoder stack:
   - masked self-attention over generated target prefix
   - cross-attention over encoder memory
6. Decoder output → linear projection to vocab logits → softmax → next-token distribution
7. Append predicted token → repeat until end token

---

## 2. Tokenization and tensors
### 2.1 Tokenization
- Input text is split into **tokens** by a tokenizer (e.g., BPE / WordPiece).
- Each token is mapped to an integer **token ID** in a vocabulary of size *V*.

### 2.2 Typical tensor shapes
Let:
- *B*: batch size
- *n*: source length
- *m*: target length
- $d_model$ : model width (hidden size)
- *h*: number of attention heads
- $d_k = d_v = d_model/h$

Common tensors:
- embedding table: $(V,d_model)$
- source IDs: $(B,n)$
- target IDs: $(B,m)$
- source embeddings: $(B,n,d_model)$
- target embeddings: $(B,m,d_model)$
- encoder output memory: $(B,n,d_model)$

---

## 3. Input X = Token Embedding + Positional Encoding
### 3.1 Token embedding
Selected from a learned embedding table.

### 3.2 Positional encoding 
Transformers have no recurrence/convolution, so order must be injected.  
Sinusoidal positional encoding is commonly used:
- $ PE(pos,2i)=\sin(pos/10000^{2i/d_model}) $
- $ PE(pos,2i+1)=\cos(pos/10000^{2i/d_model}) $

Input to the first encoder/decoder layer:
$$ X = \mathrm{TokEmbed} + \mathrm{PosEnc} $$

---

## 4. Self-Attention
queries **Q**, keys **K**, values **V** is generated 
through linear transformations from X:
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Interpretation:
- $ QK^\top $: similarity scores between each query position and each key position
- divide by $ \sqrt{d_k} $: stabilizes softmax by preventing overly large dot products
- softmax: row-wise normalization into attention weights
- multiply by **V**: weighted sum of value vectors

---

## 5. Multi-head attention (MHA)
Use *h* parallel heads with different projections:
- $\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O$
- $\mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)$

Each head operates in a lower-dimensional subspace:
- $ d_k=d_v=d_model/h $

Benefits:
- different heads attend to different relations (local vs long-range, syntactic vs semantic, etc.)
- avoids a single attention map “over-averaging” everything

---

## 6. Add & Norm (Residual + LayerNorm)
Each sublayer (attention or FFN) is wrapped by:
- $ \mathrm{LayerNorm}(x+\mathrm{Sublayer}(x)) $

Notes:
- residual connection helps optimization of deep stacks
- layer normalization stabilizes training
- Many modern implementations use **Pre-LN** (normalize before the sublayer) for better stability, but the classic paper describes the Post-LN form above.

---

## 7. Position-wise Feed-forward Network (FFN)
Each layer includes an FFN applied independently to each position:
- $ \mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2 $

Properties:
- same weights for all positions within a layer
- increases nonlinear capacity beyond attention’s linear mixing
- classic sizes: inner dimension $ d_ff $ (often larger than $ d_model $)

---

## 8. Encoder Workflow
### 8.1 One Layer 
Given layer input $X \in (B,n,d_model)$:

1. Self-attention (bidirectional, no causal mask):
- compute $ Q,K,V $ from $ X $ via linear projections
- apply MHA to mix information across all source positions

2. Add & Norm:
- residual add: $ X \leftarrow X + \mathrm{MHA}(\cdot) $
- layer norm around the sublayer

3. FFN:
- apply FFN independently to each of the *n* positions

4. Add & Norm:
- residual add and layer norm

### 8.2 Encoder stack
Stack *N* identical encoder layers:
- output is the encoder memory $ C \in (B,n,d_model) $
- every source position has a contextual representation that can attend to all source tokens

---

## 9. Decoder Workflow
Decoder generates the target sequence autoregressively.

### 9.1 Teacher forcing
- Input to decoder is the ground-truth target sequence shifted right by one:
  - prepend a begin token (e.g., `<Begin>`)
  - ensure position *i* predicts the true token at position *i*
- A causal mask ensures position *i* cannot attend to positions > *i*.

### 9.2 One decoder layer (encoder–decoder Transformer)
Given decoder layer input $ Y \in (B,m,d_model) $ and encoder memory $C \in (B,n,d_model)$:

1. Masked self-attention:
- apply MHA over $Y$ with a **causal mask**
- prevents looking at future target tokens

2. Add & Norm

3. Cross-attention (encoder–decoder attention):
- queries from decoder states
- keys/values from encoder memory $ C $
- lets each decoder position attend over all source positions

4. Add & Norm

5. FFN

6. Add & Norm

### 9.3 Decoder-only Variant(GPTs)
- remove cross-attention
- keep masked self-attention + FFN per layer

---

## 10. Masking
### 10.1 Padding Mask
If sequences are padded to a fixed length:
- padding tokens must not contribute to attention
- implement by setting attention logits for PAD positions to a large negative value before softmax

### 10.2 Causal mask
Goal: preserve autoregressive property:
- token at position *i* may attend only to positions $\le i$
Implementation pattern:
- set logits for illegal connections (future positions) to $-\infty$ before softmax
- after softmax, those positions have probability 0

---

## 11. Output Projection, Softmax, and Weight Tying
### 11.1 From Hidden States to Vocabulary Logits
Decoder output at each position is projected to vocabulary size:
- logits shape: $(B,m,V)$
- probabilities: softmax over the last dimension

### 11.2 Common Weight Tying
A frequent design in classic Transformer:
- share weights between input embedding matrices and the pre-softmax linear projection
- sometimes scale embeddings by $\sqrt{d_model}$

---

## 12. Training Objective and Regularization 
### 12.1 Objective
- cross-entropy loss for next-token prediction:
  - maximize probability of the correct next token at every target position

### 12.2 Regularization
Common techniques:
- dropout applied to sublayer outputs and to embedding+positional sums
- label smoothing (encourages less overconfident distributions)

---

## 13. Inference workflow (autoregressive decoding)
### 13.1 Greedy decoding
Repeat:
1. feed current prefix to decoder
2. compute next-token distribution
3. pick argmax token
4. append token to prefix
Stop at end token or max length.

### 13.2 Beam search
Maintain multiple best partial hypotheses (beam size *K*), expand each step, and keep top *K* by score (often with length penalty), common in translation.

### 13.3 KV cache 
During autoregressive decoding:
- cache past keys/values for each layer
- each new step reuses cached K,V, avoiding recomputation for the entire prefix

---

## 14. Complexity and key exam points
### 14.1 Dense attention cost
For sequence length *n*:
- attention time per layer is typically $O(n^2)$
- attention matrix memory is typically $O(n^2)$

### 14.2 Parallel
- encoder self-attention over all positions can be computed in parallel (no recurrence)
- training decoder with teacher forcing is also parallel over positions (mask enforces causality)
