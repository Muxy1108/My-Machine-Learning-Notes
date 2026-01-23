# Transformer Workflow Notes

## 1. Overview
A Transformer processes a token sequence by:  
1. converting tokens to vectors (embeddings),  
2. mixing information across positions using attention,  
3. refining each position with a feed-forward network (FFN),   
repeated for multiple layers.

Main architectures:
- **Encoder-only**: representation/understanding
- **Decoder-only**: autoregressive generation
- **Encoder–Decoder**: conditional generation (seq2seq)

---

## 2. Input: Text → Model Vectors
### 2.1 Tokenization
Raw text → tokens → token IDs (integers).

### 2.2 Token Embedding Lookup
Each token ID selects a vector from an embedding matrix.
- Embedding matrix shape: $(V,d)$ (vocab size *V*, hidden size *d*)
- Output sequence matrix shape: $(n,d)$ (sequence length *n*)

### 2.3 Add Positional Information
Transformer needs order information:
- **Absolute position embeddings** (add to token embeddings), or
- **relative/rotary** position methods (applied inside attention)

Typical input to layer 1:
- **X0 = token_emb + pos_emb** (shape $(n,d)$)

---

## 3. Scaled Dot-Product Attention
Given:
- Queries **Q**, Keys **K**, Values **V**

Attention:
- $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$

Interpretation:
- **QKᵀ**: similarity scores between positions
- **softmax**: turns scores into weights per query position
- **…V**: weighted sum of value vectors

Masks:
- **Padding mask**: ignores padding tokens
- **Causal mask** (decoder self-attn): blocks future positions

---

## 4. Multi-Head Attention (MHA)
Instead of one attention, use *h* heads with different projections:
- $Q_i=XW_i^Q,\;K_i=XW_i^K,\;V_i=XW_i^V$
- $\mathrm{head}_i=\mathrm{Attention}(Q_i,K_i,V_i)$
- $\mathrm{MHA}=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O$

Why heads help:
- different heads can capture different relationships (local, long-range, syntactic, semantic).

---

## 5. FFN (Position-Wise MLP)
Applied independently at each position:
- $\mathrm{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2$

Purpose:
- adds nonlinearity and capacity beyond attention’s linear mixing.

---

## 6. Encoder Workflow (Bidirectional)
### 6.1 One Encoder Layer (typical Pre-LN sketch)
For layer input **X**:  
1. $A=\mathrm{MHA}(\mathrm{LN}(X),\mathrm{LN}(X),\mathrm{LN}(X))$ (self-attn, no causal mask)  
2. $X=X+A$ (residual)  
3. $F=\mathrm{FFN}(\mathrm{LN}(X))$  
4. $X=X+F$ (residual)

After *L* layers:
- Output **H** (contextual representations), shape $(n,d)$

Use cases:
- classification, tagging, retrieval embeddings, extractive QA.

---

## 7. Decoder Workflow (Autoregressive)
### 7.1 Masked Self-Attention (Causal)
Decoder self-attention uses a **causal mask** so position *t* only attends to 1..*t*.

### 7.2 One Decoder Layer (Decoder-only)
For layer input **Y**:  
1. $A=\mathrm{MHA}(\mathrm{LN}(Y),\mathrm{LN}(Y),\mathrm{LN}(Y))$ with **causal mask**  
2. $Y=Y+A$  
3. $F=\mathrm{FFN}(\mathrm{LN}(Y))$  
4. $Y=Y+F$  

Output logits (next-token scores):
- $\mathrm{logits}=YW_{\mathrm{vocab}}$ (per position)

Training:
- teacher forcing (predict next token at each position)

Inference:
- generate token-by-token using the prefix.

---

## 8. Encoder–Decoder Workflow (Seq2Seq)
- **Encoder**: source tokens → **H**
- **Decoder**: target prefix → masked self-attn → **cross-attn** over **H** → next token

Cross-attention:
- Queries from decoder states
- Keys/values from encoder output **H**

This is effective for translation/summarization where output must stay grounded in the input.

---

## 9. KV Cache (Decoder Inference Acceleration)
During autoregressive decoding:
- cache past **K,V** so each new step reuses history instead of recomputing from scratch.

---

## 10. Complexity (Dense Attention)
For sequence length *n* (standard attention):
- time per layer: $O(n^2)$
- attention memory (weights): $O(n^2)$

This motivates long-context / efficient-attention variants.

---

## 11. Minimal End-to-End Workflow Sketch
### Encoder-only
- tokens → embeddings+pos → [Encoder layers] → **H** → task head

### Decoder-only
- prefix → embeddings+pos → [Decoder layers (causal)] → logits → next token → append → repeat

### Encoder–Decoder
- source → Encoder → **H**
- target prefix → Decoder (causal + cross-attn to **H**) → logits → generate
