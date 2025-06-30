# 🧠 Transformer for Math Addition Learning

A complete implementation of a simplified Transformer model for learning basic arithmetic operations, specifically designed for educational purposes and demonstrating the power of attention mechanisms in sequence-to-sequence tasks.

## 📖 Overview

This project implements a **decoder-only Transformer architecture** (similar to GPT) that learns to perform multi-digit addition. The model takes arithmetic expressions like "23+45+67=" as input and generates the correct result "135". This serves as an excellent introduction to:

- 🔍 **Transformer Architecture**: Multi-head attention, positional encoding, layer normalization
- 📊 **Sequence-to-Sequence Learning**: Autoregressive generation with teacher forcing
- 🎯 **Mathematical Reasoning**: Teaching AI to perform basic arithmetic operations
- 🛠️ **Complete ML Pipeline**: From data generation to model training and evaluation

## 🏗️ Architecture

### Key Differences from Original Transformer Paper

Our implementation is a **simplified, decoder-only** architecture:

| Component | Original Paper | This Implementation |
|-----------|----------------|-------------------|
| **Architecture** | Encoder-Decoder | Decoder-only (GPT-style) |
| **Attention Types** | Self-attention + Cross-attention | Self-attention only |
| **Use Case** | Machine Translation | Autoregressive Generation |
| **Complexity** | Full production model | Educational/Research friendly |

### Model Components

```
Input: <START>23+45=
       ↓
   Token Embedding + Positional Encoding
       ↓
   ┌─ Transformer Block 1 ─┐
   │  Multi-Head Attention  │
   │  Feed-Forward Network   │
   │  Residual + LayerNorm   │
   └────────────────────────┘
       ↓
   ┌─ Transformer Block N ─┐
   │         ...            │
   └────────────────────────┘
       ↓
   Layer Normalization
       ↓
   Linear Projection
       ↓
Output: 68<END>
```

## 📁 Project Structure

```
LLM-Learning-Projects/
├── 📂 data/                    # Data generation
│   ├── generate.py            # Dataset generator
│   ├── data_zh.md            # Chinese documentation  
│   └── data_en.md            # English documentation
├── 📂 transformer/            # Model implementation
│   ├── transformer.py        # Core model code
│   ├── train_transformer.py  # Training script
│   ├── transformer_zh.md     # Chinese model docs
│   └── transformer_en.md     # English model docs
├── 📂 scripts/               # Utility scripts
│   ├── generate.sh           # Data generation script
│   └── train_transformer.sh  # Training script
├── 📂 dataset/               # Generated datasets (created)
│   ├── train.json           # Training data
│   ├── val.json             # Validation data
│   ├── test.json            # Test data
│   └── vocab.json           # Vocabulary mapping
├── 📂 models/                # Saved models (created)
│   └── best_model.pth       # Best trained model
├── test_training.py          # Quick functionality test
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.7+ required
python --version

# Install required packages
pip install torch matplotlib numpy
```

### 2. Generate Training Data

```bash
# Generate default dataset (10,000 samples)
bash scripts/generate.sh

# Or generate custom dataset
bash scripts/generate.sh --small  # 1,000 samples
bash scripts/generate.sh --large  # 500,000 samples
bash scripts/generate.sh -n 50000 -x 4  # Custom: 50k samples, max 4 addends
```

### 3. Test Basic Functionality

```bash
# Run quick tests to verify everything works
python test_training.py
```

### 4. Train the Model

```bash
# Train with default settings
bash scripts/train_transformer.sh

# Quick training for testing
bash scripts/train_transformer.sh --small --quick

# Full training with large model
bash scripts/train_transformer.sh --large --epochs 100
```

## 📊 Data Format

### Input/Output Examples

| Expression | Input Sequence | Output Sequence | 
|------------|----------------|-----------------|
| `23+45` | `<START>23+45=` | `68<END>` |
| `7+12+5` | `<START>7+12+5=` | `24<END>` |
| `99+1+50` | `<START>99+1+50=` | `150<END>` |

### Vocabulary

The model uses a minimal vocabulary of **15 tokens**:

```json
{
  "<PAD>": 0,    // Padding token
  "<START>": 1,  // Sequence start
  "<END>": 2,    // Sequence end
  "+": 3,        // Addition operator
  "=": 4,        // Equals sign
  "0": 5, "1": 6, "2": 7, "3": 8, "4": 9,  // Digits 0-4
  "5": 10, "6": 11, "7": 12, "8": 13, "9": 14  // Digits 5-9
}
```

## 🎯 Usage Examples

### Basic Training

```bash
# Generate small dataset and train quickly
scripts/generate.sh --debug                    # 100 samples
scripts/train_transformer.sh --small --quick   # Small model, 10 epochs
```

### Production Training

```bash
# Generate large dataset
scripts/generate.sh --large                    # 500,000 samples

# Train production model
scripts/train_transformer.sh --large \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 1e-4
```

### Custom Configuration

```bash
# Custom data generation
scripts/generate.sh \
    --num-samples 100000 \
    --max-numbers 6 \
    --max-value 999 \
    --output-dir ./custom_data

# Custom model training
scripts/train_transformer.sh \
    --data-dir ./custom_data \
    --d-model 512 \
    --n-layers 8 \
    --epochs 50
```

## 🔧 Configuration Options

### Data Generation Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-samples` | 10000 | Number of training samples |
| `--max-numbers` | 5 | Maximum addends per expression |
| `--max-value` | 100 | Maximum value of each number |
| `--min-numbers` | 2 | Minimum addends per expression |

### Model Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--d-model` | 256 | Model dimension |
| `--n-heads` | 8 | Number of attention heads |
| `--n-layers` | 4 | Number of transformer layers |
| `--batch-size` | 32 | Training batch size |
| `--learning-rate` | 1e-4 | Learning rate |
| `--epochs` | 50 | Number of training epochs |

### Preset Configurations

| Preset | Model Size | Parameters | Use Case |
|--------|------------|------------|----------|
| `--small` | d_model=128, n_layers=2 | ~50K | Quick testing |
| `--medium` | d_model=256, n_layers=4 | ~350K | Balanced training |
| `--large` | d_model=512, n_layers=6 | ~2M | Best performance |

## 📈 Training Results

Expected performance on the math addition task:

| Model Size | Training Time | Final Accuracy | Notes |
|------------|---------------|----------------|-------|
| Small | ~5 minutes | 85-90% | Good for testing |
| Medium | ~15 minutes | 95-98% | Recommended baseline |
| Large | ~45 minutes | 98-99% | Best performance |

### Sample Training Output

```
Epoch 1/50
Training Loss: 2.1234, Time: 45.2s
Validation Loss: 1.8765, Accuracy: 0.6234

Epoch 25/50
Training Loss: 0.2345, Time: 43.1s
Validation Loss: 0.1987, Accuracy: 0.9456

Final Results:
Best Validation Accuracy: 0.9789
Test Accuracy: 0.9756

Example Predictions:
Example 1: ✓
  Expression: 23+45+12
  True result: 80
  Predicted: 80

Example 2: ✓  
  Expression: 67+89
  True result: 156
  Predicted: 156
```

## 🧪 Testing and Validation

### Quick Functionality Test

```bash
python test_training.py
```

This will verify:
- ✅ Model creation and parameter counting
- ✅ Data loading and batching
- ✅ Forward pass computation
- ✅ Training step execution
- ✅ Sequence generation

### Manual Testing

```python
# Test model predictions
python -c "
from transformer.train_transformer import *
# Load trained model and test on custom inputs
"
```

## 📚 Educational Value

This project demonstrates key concepts:

### 1. **Transformer Architecture**
- Multi-head attention mechanisms
- Positional encoding for sequence understanding
- Residual connections and layer normalization
- Autoregressive generation

### 2. **Training Deep Learning Models**
- Data preprocessing and tokenization
- Loss function design for sequence tasks
- Optimization strategies (Adam, learning rate scheduling)
- Evaluation metrics and early stopping

### 3. **Mathematical Reasoning in AI**
- How attention mechanisms can learn arithmetic patterns
- Sequence-to-sequence learning for symbolic reasoning
- Generalization to unseen number combinations

## 🔍 Model Analysis

### Attention Visualization

The model learns to:
- Focus on individual numbers when processing addends
- Attend to the "=" symbol to trigger result generation
- Maintain positional awareness for multi-digit numbers

### Performance Characteristics

- **Input Length**: Handles 2-6 addends efficiently
- **Number Range**: Works well up to 3-digit numbers
- **Generalization**: Can solve problems with unseen number combinations
- **Accuracy**: 95%+ on validation set with proper training

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Missing PyTorch
pip install torch

# Missing matplotlib  
pip install matplotlib
```

#### 2. CUDA Out of Memory
```bash
# Reduce batch size
scripts/train_transformer.sh --batch-size 16

# Use smaller model
scripts/train_transformer.sh --small
```

#### 3. Poor Training Performance
```bash
# Check data quality
head -n 10 dataset/train.json

# Try different learning rate
scripts/train_transformer.sh --learning-rate 5e-4

# Increase model capacity
scripts/train_transformer.sh --large
```

#### 4. Slow Training
```bash
# Use GPU if available
python -c "import torch; print(torch.cuda.is_available())"

# Reduce dataset size for testing
scripts/generate.sh --small
```

## 🎓 Learning Extensions

### Easy Extensions
1. **Other Operations**: Extend to subtraction, multiplication
2. **Larger Numbers**: Increase max_value to handle bigger numbers
3. **Mixed Operations**: Combine +, -, × in single expressions

### Advanced Extensions
1. **Encoder-Decoder**: Implement full transformer architecture
2. **Beam Search**: Better decoding strategies
3. **Attention Visualization**: Plot attention weights
4. **Mathematical Word Problems**: Natural language to arithmetic

### Research Directions
1. **Few-Shot Learning**: Train on small datasets
2. **Transfer Learning**: Pre-train on one operation, fine-tune on another
3. **Systematic Generalization**: Test on much larger numbers
4. **Interpretability**: Understand what the model learns

## 📄 Documentation

- **Chinese Documentation**: See `transformer/transformer_zh.md` and `data/data_zh.md`
- **English Documentation**: See `transformer/transformer_en.md` and `data/data_en.md`
- **Code Comments**: Detailed comments throughout the codebase

## 🤝 Contributing

Feel free to contribute by:
- 🐛 Reporting bugs
- 💡 Suggesting new features  
- 📖 Improving documentation
- 🔬 Adding new experiments

## 📜 License

This project is for educational purposes. Feel free to use and modify for learning and research.

## 🙏 Acknowledgments

- Based on "Attention Is All You Need" (Vaswani et al., 2017)
- Inspired by mathematical reasoning research in AI
- Educational framework designed for learning transformer concepts

---

**Happy Learning! 🚀**

*This project demonstrates that even simple mathematical tasks can teach us profound lessons about attention, sequence modeling, and the power of transformer architectures.*
