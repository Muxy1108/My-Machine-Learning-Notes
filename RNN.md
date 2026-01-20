# Common RNN Categories

> by **Architecture**  
> A separate orthogonal dimension is the **recurrent cell type** (Vanilla / LSTM / GRU), which can be combined with any architecture below.

---

## 0. Two Orthogonal Axes

### A. Architecture
- Unidirectional (Vanilla / Forward)
- Bidirectional (BiRNN)
- Stacked / Deep RNN
- Encoder–Decoder (Seq2Seq, often with Attention)

### B. Cell (recurrent unit) types
- **Vanilla RNN cell**:

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

- **LSTM cell**: gated memory cell $c_t$
- **GRU cell**: update/reset gates

---

## 1. Vanilla RNN (Basic / Elman RNN, Unidirectional)

### 1.1 Definition
Processes a sequence from $1 \to T$; the state at time $t$ depends only on the past:

$$
h_t = f(W_x x_t + W_h h_{t-1} + b)
$$

Typically $f=\tanh$ (classic) or ReLU (less common in basic RNNs).

### 1.2 Common input–output patterns
- **Many-to-one** (sequence classification): use $h_T$ (or pooling over $\{h_t\}$) to predict $y$
- **Many-to-many (aligned)** (sequence labeling): output $y_t$ at each step

### 1.3 Pros / cons
- Pros: simple; streaming/online friendly  
- Cons: struggles with long-range dependencies (vanishing/exploding gradients); often replaced by LSTM/GRU in practice

---

## 2. Bidirectional RNN (BiRNN / BiLSTM / BiGRU)

### 2.1 Definition
Runs two RNNs over the same sequence:

- Forward direction:

$$
\overrightarrow{h}_t = f(x_t,\overrightarrow{h}_{t-1})
$$

- Backward direction:

$$
\overleftarrow{h}_t = f(x_t,\overleftarrow{h}_{t+1})
$$

Combine them (most commonly by concatenation):

$$
h^{bi}_t = [\overrightarrow{h}_t;\overleftarrow{h}_t]
$$

### 2.2 When it is useful
Tasks where a token/frame label depends on **both left and right context**, e.g. sequence tagging (POS/NER), offline speech frame classification, offline text encoding.

### 2.3 Limitation
Not suitable for strict real-time/online settings.

---

## 3. Stacked / Deep RNN

### 3.1 Definition
Multiple recurrent layers stacked vertically; layer $\ell$ receives the hidden sequence from layer $\ell-1$:

$$
h_t^{(\ell)} = f\!\left(h_t^{(\ell-1)},\, h_{t-1}^{(\ell)}\right),\quad h_t^{(0)}=x_t
$$

### 3.2 Intuition
- Lower layers tend to capture more local/short-range patterns
- Higher layers tend to capture more abstract/global structure (empirical tendency)

### 3.3 Pros / cons
- Pros: higher representational capacity  
- Cons: harder to optimize, slower, more prone to overfitting  
- Common tricks: dropout (including inter-layer), residual/skip connections, normalization (implementation-dependent), gradient clipping  

---

## 4. Encoder–Decoder RNN (Seq2Seq)

### 4.1 Problem setting
Maps an input sequence $x_{1:T}$ to an output sequence $y_{1:U}$ (lengths may differ), e.g. machine translation, summarization, dialogue generation, early attention-based ASR.

### 4.2 Basic Seq2Seq (without Attention)
Encoder:

$$
h_t^{enc}=f(x_t,h_{t-1}^{enc})
$$

Context vector (classic choice):

$$
c=h_T^{enc}
$$

Decoder:

$$
h_u^{dec} = f\!\left(y_{u-1},\, h_{u-1}^{dec},\, c\right),\quad
p(y_u \mid y_{1:u-1}, x) = \mathrm{softmax}\!\left(W h_u^{dec}\right)
$$



**Bottleneck:** compressing the whole input into a fixed vector $c$ loses information for long sequences.

### 4.3 Seq2Seq with Attention (modern RNN-Seq2Seq)
At each decoder step $u$, compute attention weights over all encoder states:

$$
\alpha_{u,t}=\text{softmax}(\text{score}(h_u^{dec},h_t^{enc})),\quad
c_u=\sum_{t=1}^{T}\alpha_{u,t}h_t^{enc}
$$

Use $[h_u^{dec};c_u]$ to predict $y_u$.

Benefits: mitigates the fixed-vector bottleneck, improves long-sequence performance, provides alignment interpretability.

### 4.4 to Transformer
Transformer retains the encoder–decoder paradigm but replaces recurrence with (self-)attention, enabling better parallelism and long-range modeling.

---

