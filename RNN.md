# RNN

Classification:  
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

## 1A. Vanilla RNN (Basic / Elman RNN, Unidirectional)

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

## 2A. Bidirectional RNN (BiRNN / BiLSTM / BiGRU)

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

## 3A. Stacked / Deep RNN

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

## 4A. Encoder–Decoder RNN (Seq2Seq)

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

## 1B. LSTM

LSTM maintains:
- **cell state** *c*: long-term memory “highway”
- **hidden state** *h*: exposed / working state

This separation makes long-range credit assignment easier.

### 1.1 Equations
Given input *x* at time *t*, previous states $h_{t-1}$, $c_{t-1}$:  

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

Where:
- *f* is the **forget gate**: how much old memory to keep
- *i* is the **input gate**: how much new content to write
- *o* is the **output gate**: how much memory to expose
- *c̃* is the **candidate memory**
- ⊙ is elementwise multiplication
- σ is sigmoid

### 1.2 Gates State
- *f* ≈ 1 and *i* ≈ 0: keep memory, do not overwrite
- *f* ≈ 0 and *i* ≈ 1: erase old memory, write new memory
- *o* ≈ 0: memory exists but not revealed in *h*

### 1.3 Long-term Dependency
The update of *c* has an additive path:

$$c_t = f_t \odot c_{t-1} + \text{(new write)}$$

If *f* stays near 1, information (and gradients) can flow through many time steps with less attenuation than in vanilla RNN.

### 1.4 Common Variants    
- **Peephole LSTM**: gates also look at *c*
- **Coupled input-forget gate**: reduce parameters by tying *i* and *f*

## 2B. GRU

GRU merges “cell” and “hidden” into a single state *h*.
It uses two gates:
- **update gate** *z*: keep old vs accept new
- **reset gate** *r*: how much past to use when forming candidate

### 2.1 Equations

$$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$
$$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$$
$$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Interpretation:
- *z* controls the interpolation between old state and new candidate
- *r* controls whether we “reset” past when computing the candidate

### 2.3 Gates State
- *z* ≈ 0: keep old state, slow update, long memory
- *z* ≈ 1: mostly replace by new candidate, fast adaptation
- *r* ≈ 0: candidate ignores old state (useful at boundaries/topic shifts)
- *r* ≈ 1: candidate fully uses old state