# CNN

## 1. Definition
A **Convolutional Neural Network (CNN)** is a deep neural network designed to process data with **grid-like structure**, especially:
- Images (2D grids of pixels)
- Audio spectrograms (2D)
- Time series / signals (1D)
- Videos / volumetric data (3D)

---

## 2. Inductive Biases

- **Local connectivity**: nearby pixels/timestamps are more correlated than far ones.
- **Weight sharing**: the same pattern (e.g., an edge) can appear anywhere.
- **Translation equivariance**: shifting the input shifts the feature map similarly.

These biases reduce parameters and improve generalization compared to plain MLPs on images.

---

## 3. Structure

### 3.1 Convolution layer
**Input**: $\(X \in \mathbb{R}^{H \times W \times C_{in}}\)$  
**Kernel/weights**: $\(W \in \mathbb{R}^{K_h \times K_w \times C_{in} \times C_{out}}\)$  
**Output**: $\(Y \in \mathbb{R}^{H' \times W' \times C_{out}}\)$

Computation:
$\(Y_{i,j,c}=\sum_{u,v}\sum_{d=1}^{C_{in}} W_{u,v,d,c}\,X_{i+u,j+v,d}+b_c\)$

Notes:
- Deep learning frameworks often implement **cross-correlation** (kernel not flipped), still called “convolution”.

### 3.2 Activation functions
Adds nonlinearity.
- ReLU: $\(\phi(x)=\max(0,x)\)$
- GELU, LeakyReLU, etc.

### 3.3 Normalization
Common in CNNs to stabilize and accelerate training.
- **BatchNorm**: normalizes per-channel using batch statistics (typical for ConvNets)
- **GroupNorm / LayerNorm**: useful when batch size is small

### 3.4 Downsampling
Purpose:
- Reduce spatial size $\(H,W\)$ to save compute/memory
- Increase receptive field
- Add some translation invariance

Methods:
- Max/Avg Pooling
- Strided convolution (stride \> 1) (common in modern CNNs)

### 3.5 Head (task-specific output)
- Classification: Global Average Pooling (GAP) + Linear
- Detection: bounding boxes + classes
- Segmentation: per-pixel predictions (often needs upsampling/decoder)

---

## 4. Convolution hyperparameters and output size

### 4.1 Kernel size (K)
Common: $\(3\times 3\)$, $\(1\times 1\)$, sometimes larger (e.g.,  $\(7\times 7\)$ ).

### 4.2 Stride (S)
- $\(S=1\)$: usually keeps resolution (with padding)
- $\(S=2\)$: downsample by ~2

### 4.3 Padding (P)
- "same" padding: output size roughly preserved
- "valid" padding: no padding, output shrinks

Output shape formula:
$\(H'=\left\lfloor \frac{H+2P-K_h}{S}\right\rfloor+1\)$,
$\(W'=\left\lfloor \frac{W+2P-K_w}{S}\right\rfloor+1\)$

### 4.4 Dilation (D)
Increases receptive field without increasing parameters much (common in segmentation).

---

## 5. Receptive field (RF)
The **receptive field** of a unit is the region in the input that influences it.
- Deeper layers generally have larger RF.
- Downsampling and dilation increase RF faster.

Intuition:
- Shallow layers: edges, corners, textures
- Deeper layers: parts, shapes, objects (semantic features)

---

## 6. Common CNN operator variants

### 6.1 Standard Convolution
Full spatial + channel mixing in one operation.

### 6.2 Pointwise Convolution (*1 × 1*)
- Mixes channels only
- Used for bottlenecks, channel expansion/reduction, feature fusion

### 6.3 Group Convolution
Channels split into groups; each group convolved separately.
- Reduces compute
- Used in ResNeXt

### 6.4 Depthwise Separable Convolution
Factorizes standard conv into:
1) **Depthwise**: per-channel spatial convolution  
2) **Pointwise**: $\(1\times 1\)$ channel mixing  
Used in MobileNet; huge efficiency gain.

### 6.5 Dilated (Atrous) Convolution
Larger RF without pooling/strides (preserves resolution).

### 6.6 Transposed Convolution (Deconvolution)
Used for learnable upsampling; can cause checkerboard artifacts.
Often replaced by: Upsample (interpolation) + Conv.

### 6.7 Deformable Convolution
Learns offsets for sampling positions; better for geometric variations.

---

## 7. Backbone Families

### 7.1 VGG-style
- Many stacked $\(3\times 3\)$ convs + pooling
- Simple but heavy (lots of params)

### 7.2 Inception-style
- Multi-branch blocks with different kernel sizes
- Multi-scale feature extraction

### 7.3 ResNet
Core idea:
$y = F(x) + x$

Benefits:
- Enables very deep networks
- Improves optimization (reduces degradation)

### 7.4 DenseNet
- Dense connectivity: each layer receives all previous features
- Strong feature reuse; can be memory-heavy

### 7.5 Efficient / Mobile CNNs
- MobileNet, ShuffleNet, EfficientNet
- Use separable/group conv, bottlenecks, scaling rules

### 7.6 Modern CNNs (e.g., ConvNeXt)
- CNNs redesigned with training and block choices inspired by Transformers
- Competitive performance with ViT in many tasks

---

## 8. CNNs for different tasks

### 8.1 Image classification
Pipeline:
Backbone -> GAP -> Linear -> Softmax

Metrics:
- Top-1 / Top-5 accuracy

### 8.2 Object detection
Outputs bounding boxes + class labels.
Typical structure:
Backbone + multi-scale feature fusion (FPN/Neck) + detection head

### 8.3 Semantic / Instance segmentation
Per-pixel (or per-object mask) predictions.
Common structures:
- Encoder–Decoder (U-Net)
- Dilated conv + ASPP (DeepLab-style)
- Skip connections to restore detail

### 8.4 Super-resolution / restoration
Upsampling and reconstruction:
- Residual blocks
- PixelShuffle / upsample+conv

---

## 9. Training and engineering notes

### 9.1 Regularization
- Data augmentation (flip, crop, color jitter; MixUp/CutMix)
- Weight decay
- Dropout (often in heads; less common inside classic Conv blocks)

### 9.2 Optimization
- SGD + momentum (classic for CNNs)
- AdamW (common in modern setups)
- Learning rate schedule matters a lot (cosine, step, warmup)

### 9.3 Compute / memory considerations
Compute roughly scales with:
$\(H \cdot W \cdot K_h \cdot K_w \cdot C_{in} \cdot C_{out}\)$

Common speedups:
- Reduce resolution earlier (stride/pooling)
- Use separable/group conv
- Bottleneck designs $(\(1\times 1\) reduce -> \(3\times 3\) -> \(1\times 1\) expand)$
- Quantization / pruning for deployment

---

