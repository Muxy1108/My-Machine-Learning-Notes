# GNN

## 1. Why GNN
CNN assumes **grid-like** structure (images), RNN assumes **sequence** structure (time series).
Many real-world data are **non-Euclidean** and naturally graphs:
- Social networks (users ↔ relations)
- Molecules (atoms ↔ bonds)
- Recommenders (user–item bipartite graph)
- Knowledge graphs (entities ↔ relations)
- Traffic networks, citations, web links, circuits, etc.

Prediction depends heavily on **connectivity + local substructure**.

---

## 2. Core Paradigm
Most GNNs follow the **Message Passing Neural Network (MPNN)** view.

### 2.1 Layer update
At layer $\(k\)$, each node $\(v\)$ aggregates messages from neighbors:
\[
m_v^{(k)}=\text{AGG}\Big(\{\,h_u^{(k-1)}:u\in N(v)\,\}\Big)
\]
Then updates its representation:
\[
h_v^{(k)}=\text{UPDATE}\big(h_v^{(k-1)}, m_v^{(k)}\big)
\]

Where:
- $\(h_v^{(0)} = x_v\)$ (initial embedding from node features)
- **AGG** must be **permutation-invariant** (order of neighbors should not matter): sum/mean/max/attention
- With $\(K\)$ layers, node representation contains up to **K-hop** neighborhood information


---

## 3. Tasks

### 3.1 Node-level tasks
- Node classification / regression
- Example: classify papers by topic in a citation graph

### 3.2 Edge / link-level tasks
- Link prediction, recommendation, relation prediction
- Example: predict whether user $\(u\)$ will interact with item $\(v\)$

### 3.3 Graph-level tasks
- Whole-graph classification / regression
- Example: molecular property prediction

Graph-level usually needs a **readout / pooling**:
\[
h_G=\text{READOUT}(\{h_v^{(K)}\}_{v\in V})
\]
READOUT examples: sum/mean pooling, attention pooling, Set2Set

---

## 4. Common Families
### 4.1 GCN (Graph Convolutional Network)
- “Convolution” generalized as normalized neighbor averaging
- Often uses adjacency normalization to stabilize training

### 4.2 GraphSAGE
- Designed for **large graphs** via **neighbor sampling**
- Works well for **inductive** settings (generalize to unseen nodes/graphs)

### 4.3 GAT (Graph Attention Network)
- Uses attention to learn different importance weights for neighbors
- Useful when not all neighbors are equally informative

### 4.4 GIN (Graph Isomorphism Network)
- Stronger ability to distinguish graph structures
- Often used in graph-level tasks and graph benchmarks

### 4.5 Variants / Extensions
- R-GCN: multiple relation types (heterogeneous / knowledge graphs)
- GGNN: gated updates (GRU-like)
- MPNN in chemistry: explicit edge/bond message passing

---

## 5. Training Settings

### 5.1 Transductive vs Inductive
- **Transductive**: full graph seen during training, some nodes unlabeled; test nodes are in the same graph
- **Inductive**: model must generalize to new nodes/graphs not seen during training

### 5.2 Scalability on Large Graphs
Full-batch message passing can be expensive ( $\(O(|E|)\)$ ).
Common solutions:
- Neighbor sampling (GraphSAGE-style minibatch)
- Subgraph/cluster training (Cluster-GCN)
- Precomputation / caching for fixed features

---

## 6. Implementation Building Blocks
- Node embedding: linear layer / MLP over $\(x_v\)$
- Message passing layers: aggregation + update (often MLP)
- Normalization: batch norm, layer norm, graph norm
- Residual connections: help deeper models
- Dropout: regularization
- Readout/pooling (graph-level)

---

## 7. Challenges
### 7.1 Over-smoothing
With many layers, node embeddings may become too similar.
Fixes:
- Residual / skip connections
- Normalization
- Limit depth, or use Jumping Knowledge (JK) connections

### 7.2 Over-squashing
Long-range information is compressed into limited-size vectors through narrow paths.
Fixes:
- Add structural/positional encodings
- Modify graph (virtual nodes/edges)
- Use attention/diffusion mechanisms

### 7.3 Expressivity Limits
Some GNNs cannot distinguish certain non-isomorphic graphs.
Fixes:
- Stronger aggregators (e.g., GIN-like)
- Positional/structural features (Laplacian PE, random-walk features)