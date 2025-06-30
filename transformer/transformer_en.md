# Transformer Model Implementation Guide

## Table of Contents
- [Overview](#overview)
- [Core Components](#core-components)
  - [1. Multi-Head Attention](#1-multi-head-attention)
  - [2. Feed-Forward Network](#2-feed-forward-network)
  - [3. Layer Normalization](#3-layer-normalization)
  - [4. Positional Encoding](#4-positional-encoding)
  - [5. Transformer Block](#5-transformer-block)
  - [6. Complete Transformer Model](#6-complete-transformer-model)
- [Mathematical Formulations](#mathematical-formulations)
- [Usage Instructions](#usage-instructions)
- [Model Configuration](#model-configuration)
- [Extended Features](#extended-features)

## Overview

This is a simple yet complete implementation of the Transformer model based on the "Attention Is All You Need" paper. The implementation includes all core components of the Transformer architecture and can be used for language modeling, text classification, and other natural language processing tasks.

### Key Features
- ✅ Complete multi-head attention mechanism
- ✅ Positional encoding
- ✅ Layer normalization and residual connections
- ✅ Feed-forward neural networks
- ✅ Mask mechanism support
- ✅ Configurable hyperparameters
- ✅ Batch processing support

## Core Components

### 1. Multi-Head Attention

Multi-head attention is the core component of Transformer, allowing the model to attend to information from different positions simultaneously.

#### Mathematical Formula
For input sequences, the attention mechanism is computed as follows:

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

#### Implementation Details
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        # d_model: model dimension (typically 512)
        # n_heads: number of attention heads (typically 8)
        # d_k: dimension per head = d_model // n_heads
```

#### Key Steps
1. **Linear Transformation**: Transform input through W^Q, W^K, W^V to get Query, Key, Value
2. **Multi-Head Split**: Split Q, K, V into multiple heads
3. **Scaled Dot-Product Attention**: Compute attention weights and outputs
4. **Head Concatenation**: Concatenate outputs from multiple heads
5. **Output Projection**: Final linear transformation through W^O

### 2. Feed-Forward Network

The feed-forward network provides non-linear transformation capabilities to the Transformer.

#### Mathematical Formula
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

#### Structure Characteristics
- Two-layer fully connected network
- First layer: `d_model → d_ff` (typically d_ff = 4 * d_model)
- Activation function: ReLU
- Second layer: `d_ff → d_model`
- Dropout regularization

### 3. Layer Normalization

Layer normalization is used to stabilize the training process and is applied after each sub-layer.

#### Mathematical Formula
```
LayerNorm(x) = γ * (x - μ) / σ + β
where:
μ = mean(x)  # mean
σ = std(x)   # standard deviation
γ, β are learnable parameters
```

#### Functions
- Normalize activation distributions
- Accelerate convergence
- Improve training stability

### 4. Positional Encoding

Since Transformer lacks recurrent structure, positional encoding is needed to understand sequence order.

#### Mathematical Formula
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos`: position index
- `i`: dimension index
- `d_model`: model dimension

#### Characteristics
- Fixed sine and cosine functions
- Allows the model to learn relative position relationships
- Supports sequences of arbitrary length

### 5. Transformer Block

The Transformer block is the basic building unit, containing attention and feed-forward networks.

#### Structure
```
x → Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm → output
    ↑_______________|                    ↑_______________|
    residual connection                  residual connection
```

#### Mathematical Representation
```
# Sub-layer 1: Multi-head attention
x1 = LayerNorm(x + MultiHeadAttention(x))

# Sub-layer 2: Feed-forward network  
x2 = LayerNorm(x1 + FeedForward(x1))
```

### 6. Complete Transformer Model

The complete model stacks multiple Transformer blocks and adds input/output processing.

#### Architecture Flow
1. **Input Embedding**: Convert tokens to vectors
2. **Positional Encoding**: Add positional information
3. **Transformer Blocks**: N stacked Transformer blocks
4. **Layer Normalization**: Final layer normalization
5. **Output Projection**: Project to vocabulary size

## Mathematical Formulations

### Detailed Attention Mechanism Derivation

1. **Input Representation**
   ```
   X ∈ R^(seq_len × d_model)  # input sequence
   ```

2. **Linear Transformations**
   ```
   Q = XW^Q,  W^Q ∈ R^(d_model × d_model)
   K = XW^K,  W^K ∈ R^(d_model × d_model)  
   V = XW^V,  W^V ∈ R^(d_model × d_model)
   ```

3. **Multi-Head Split**
   ```
   Q_i = Q[:, i*d_k:(i+1)*d_k]  # Query for head i
   K_i = K[:, i*d_k:(i+1)*d_k]  # Key for head i
   V_i = V[:, i*d_k:(i+1)*d_k]  # Value for head i
   ```

4. **Attention Computation**
   ```
   Attention_i = softmax(Q_i K_i^T / √d_k) V_i
   ```

5. **Head Concatenation**
   ```
   MultiHead = Concat(Attention_1, ..., Attention_h) W^O
   ```

### Positional Encoding Derivation

Positional encoding uses a combination of sine and cosine functions:

```python
# For position pos and dimension i:
if i % 2 == 0:  # even dimensions
    PE[pos][i] = sin(pos / 10000^(i/d_model))
else:           # odd dimensions  
    PE[pos][i] = cos(pos / 10000^((i-1)/d_model))
```

Advantages of this design:
- Each position has a unique encoding
- Relative position relationships can be learned through trigonometric identities
- Supports sequences longer than training length

## Usage Instructions

### 1. Basic Usage

```python
import torch
from transformer import Transformer, create_padding_mask

# Create model
model = Transformer(
    vocab_size=10000,  # vocabulary size
    d_model=512,       # model dimension
    n_heads=8,         # number of attention heads
    n_layers=6,        # number of Transformer layers
    d_ff=2048,         # feed-forward hidden dimension
    max_len=5000,      # maximum sequence length
    dropout=0.1        # dropout rate
)

# Prepare input data
batch_size = 2
seq_len = 20
input_ids = torch.randint(0, 10000, (batch_size, seq_len))

# Create mask
padding_mask = create_padding_mask(input_ids)

# Forward pass
output = model(input_ids, padding_mask)
print(f"Output shape: {output.shape}")  # [batch_size, seq_len, vocab_size]
```

### 2. Training Example

```python
import torch.optim as optim
import torch.nn as nn

# Initialize model and optimizer
model = Transformer(vocab_size=10000)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, target_ids = batch
        
        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), target_ids.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

### 3. Inference Example

```python
# Text generation
model.eval()
with torch.no_grad():
    # Input starting sequence
    input_ids = torch.tensor([[1, 2, 3]])  # [batch_size, seq_len]
    
    for _ in range(50):  # generate 50 tokens
        outputs = model(input_ids)
        
        # Get next token
        next_token = torch.argmax(outputs[:, -1, :], dim=-1, keepdim=True)
        
        # Concatenate to input sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
    print("Generated sequence:", input_ids)
```

## Model Configuration

### Standard Configurations

| Config | Base | Large |
|--------|------|-------|
| d_model | 512 | 1024 |
| n_heads | 8 | 16 |
| n_layers | 6 | 24 |
| d_ff | 2048 | 4096 |
| Parameters | ~65M | ~340M |

### Custom Configurations

```python
# Small model (for experiments)
small_config = {
    'vocab_size': 5000,
    'd_model': 256,
    'n_heads': 4,
    'n_layers': 4,
    'd_ff': 1024,
    'dropout': 0.1
}

# Large model (for production)
large_config = {
    'vocab_size': 50000,
    'd_model': 1024,
    'n_heads': 16,
    'n_layers': 12,
    'd_ff': 4096,
    'dropout': 0.1
}
```

## Extended Features

### 1. Masking Mechanisms

#### Padding Mask
```python
def create_padding_mask(seq, pad_idx=0):
    """Create padding mask to ignore padding tokens"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
```

#### Causal Mask (for decoder)
```python
def generate_square_subsequent_mask(sz):
    """Create causal mask to prevent seeing future information"""
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf'))
```

### 2. Performance Optimization

#### 1. Gradient Accumulation
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(step):
    warmup_steps = 4000
    return min(step ** -0.5, step * warmup_steps ** -1.5)

scheduler = LambdaLR(optimizer, lr_lambda)
```

### 3. Model Saving and Loading

```python
# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': model_config,
}, 'transformer_model.pth')

# Load model
checkpoint = torch.load('transformer_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Important Notes

1. **Memory Usage**: Attention mechanism has O(n²) complexity, long sequences consume significant memory
2. **Gradient Explosion**: Recommend using gradient clipping to prevent gradient explosion
3. **Learning Rate**: Transformer is sensitive to learning rate, recommend using warmup strategy
4. **Data Preprocessing**: Ensure proper tokenization and padding handling

## Performance Tips

### 1. Memory Optimization
- Use gradient checkpointing for large models
- Implement dynamic batching based on sequence length
- Consider using mixed precision training (FP16)

### 2. Training Stability
- Apply gradient clipping (typically 1.0)
- Use warmup learning rate schedule
- Monitor gradient norms during training

### 3. Inference Optimization
- Use key-value caching for autoregressive generation
- Implement beam search for better text generation
- Consider quantization for deployment

## Validation and Testing

```python
def validate_model():
    """Validate model implementation"""
    model = Transformer(vocab_size=1000, d_model=128, n_heads=4, n_layers=2)
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(input_ids)
        assert output.shape == (batch_size, seq_len, 1000)
        print("✅ Forward pass test passed")
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total trainable parameters: {total_params:,}")
    
    # Test gradient flow
    model.train()
    output = model(input_ids)
    loss = output.sum()
    loss.backward()
    
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "Some parameters don't have gradients"
    print("✅ Gradient flow test passed")

# Run validation
validate_model()
```

## References

- Vaswani, A., et al. "Attention is all you need." NIPS 2017.
- Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
- Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI Blog 2019.
- Brown, T., et al. "Language Models are Few-Shot Learners." NeurIPS 2020.
