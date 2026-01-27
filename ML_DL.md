# Machine Learning and Deep Learning 

## 1. Relationship 

$$\text{AI} \supset \text{ML} \supset \text{Representation Learning} \supset \text{DL}$$

- **Machine Learning (ML)**: Learn patterns or decision rules from data.
- **Deep Learning (DL)**: ML methods based on multi-layer neural networks that automatically learn representations.

---



## 2. Unified Training Objective: Empirical Risk Minimization

In supervised learning, many ML and DL models can be written as:

$$\theta^{*}=\arg\min_{\theta}\frac{1}{N}\sum_{i=1}^{N}\ell\!\left(f_{\theta}(x_i),y_i\right)$$

Where:

$N$: Number of samples.

$$\ell(\cdot)$$: Loss function.

$$\theta^{*}$$: Optimal parameters.

This formulation applies to linear regression, logistic regression, neural networks, etc.

---

## 4. Difference I: Feature Engineering vs Representation Learning

### 4.1 Traditional Machine Learning

Features are manually designed:

$$\hat y \approx g\!\left(w^{\top}\phi(x)\right)$$

- $\phi(x)$: hand-crafted feature mapping  
- $w$: model parameters  
- $g(\cdot)$: classifier or regressor  

Pipeline:

$$x \rightarrow \phi(x) \rightarrow \text{ML model} \rightarrow \hat y$$

Model performance strongly depends on feature quality.

---

### 4.2 Deep Learning

Deep learning learns features automatically in an end-to-end manner:

$$\hat y \approx f_{\theta}(x)$$

With a layered structure:

$$f_{\theta}(x)=f^{(L)}\!\left(f^{(L-1)}\!\left(\cdots f^{(1)}(x)\right)\right)$$

- Shallow layers: low-level patterns  
- Deep layers: high-level semantic representations  

---

## 5. Difference II: Data Scale and Model Capacity

General empirical rule:

$$\text{Model capacity} \uparrow \;\Rightarrow\; \text{Data requirement} \uparrow $$

- Small datasets: traditional ML is often more stable.
- Large-scale datasets: deep learning usually achieves higher performance.

---

## 6. Difference III: Interpretability and Engineering Cost

### Interpretability

Linear models:

$$\hat y = w^{\top}x$$

Parameters directly reflect feature importance.

Deep models:

$$\hat y = f_{\theta}(x)$$

Parameters are distributed across layers; interpretability is weaker and requires auxiliary methods.

---

### Engineering Cost

- **Traditional ML**
  - Fast training
  - Low computational cost
  - Easier deployment

- **Deep Learning**
  - Requires GPU/TPU
  - Longer training time
  - Higher engineering complexity

---

## 7. Common Machine Learning Algorithms

### Supervised Learning
- Linear Regression
- Logistic Regression
- Support Vector Machines (SVM)
- k-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Trees
- Random Forests
- Gradient Boosted Decision Trees (GBDT, XGBoost, LightGBM)

### Unsupervised Learning
- K-means Clustering
- Principal Component Analysis (PCA)

---

## 8. Practical Model Selection Guidelines

- Structured tabular data:

$$\text{GBDT / XGBoost / LightGBM}$$

- Small datasets with interpretability requirements:

$$\text{Linear or Logistic Regression}$$

- Medium-sized nonlinear classification:

$$\text{SVM with RBF kernel}$$

- Images, speech, natural language:

$$\text{Deep Neural Networks}$$

---

## 9. Summary

Traditional ML relies on **human-designed features**, while DL learns features automatically.  

Deep learning excels with large, **unstructured** data, while traditional machine learning remains highly competitive for structured data and interpretable models.
