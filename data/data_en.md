# Math Addition Data Generator

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Data Format](#data-format)
- [Usage](#usage)
- [Configuration Parameters](#configuration-parameters)
- [Transformer Integration](#transformer-integration)
- [Data Analysis](#data-analysis)
- [Extended Features](#extended-features)

## Overview

This is a mathematical addition data generator specifically designed for training Transformer models. It automatically generates large amounts of addition operation data for training sequence-to-sequence models to learn basic mathematical computation abilities.

### Core Features
- ✅ Randomly generate multi-term addition expressions
- ✅ Automatically calculate correct answers
- ✅ Build complete vocabulary
- ✅ Automatic dataset splitting (train/validation/test)
- ✅ Support configurable data scale and complexity
- ✅ Seamless integration with Transformer models

## Features

### 1. Flexible Data Generation

```python
# Generation example
Expression: "23+45+12"
Result: 80
Input sequence: "<START>23+45+12="
Output sequence: "80<END>"
```

#### Configurable Parameters
- **Number of addends**: 2-5 numbers (configurable)
- **Number range**: 1-100 (configurable)
- **Sample quantity**: Any amount
- **Random seed**: Ensure reproducible results

### 2. Complete Vocabulary Design

```json
{
  "<PAD>": 0,    // Padding token
  "<START>": 1,  // Start token
  "<END>": 2,    // End token
  "+": 3,        // Plus sign
  "=": 4,        // Equals sign
  "0": 5,        // Digits 0-9
  "1": 6,
  ...
  "9": 14
}
```

#### Vocabulary Characteristics
- Total of 15 tokens
- Support for arbitrary multi-digit numbers (through single digit combinations)
- Include necessary mathematical operators
- Special tokens for sequence marking

### 3. Intelligent Data Processing

#### Tokenization
```python
# Input: "123+45="
# Tokenization result: ['1', '2', '3', '+', '4', '5', '=']
# Token IDs: [6, 7, 8, 3, 9, 10, 4]
```

#### Duplicate Avoidance
- Automatically detect duplicate expressions
- Ensure dataset diversity
- Improve training effectiveness

### 4. Dataset Splitting

Default split ratios:
- **Training set**: 80%
- **Validation set**: 10% 
- **Test set**: 10%

Supports custom split ratios.

## Data Format

### Training Example Structure

```json
{
  "input_ids": [1, 6, 7, 3, 9, 10, 4],           // Input token sequence
  "output_ids": [11, 8, 2],                      // Output token sequence
  "expression": "12+45",                          // Original expression
  "result": 57,                                   // Calculation result
  "input_text": "<START>12+45=",                 // Input text
  "output_text": "57<END>"                       // Output text
}
```

### File Structure

```
data_output/
├── train.json      # Training set
├── val.json        # Validation set
├── test.json       # Test set
└── vocab.json      # Vocabulary
```

## Usage

### 1. Basic Usage

```bash
# Generate default dataset (10,000 samples)
python data/generate.py

# Generate specified number of samples
python data/generate.py --num_samples 50000

# Specify output directory
python data/generate.py --output_dir ./my_data
```

### 2. Advanced Configuration

```bash
# Complete configuration example
python data/generate.py \
    --num_samples 100000 \
    --max_numbers 6 \
    --max_value 999 \
    --min_numbers 2 \
    --output_dir ./large_dataset \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42
```

### 3. Programming Interface

```python
from data.generate import MathDataGenerator

# Create generator
generator = MathDataGenerator(
    max_numbers=5,
    max_value=100,
    min_numbers=2
)

# Generate dataset
dataset = generator.generate_dataset(1000)

# Split dataset
train_set, val_set, test_set = generator.split_dataset(dataset)

# Save data
generator.save_dataset(train_set, 'train.json')
generator.save_vocab('vocab.json')
```

## Configuration Parameters

### Command Line Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_samples` | 10000 | Total number of samples to generate |
| `--max_numbers` | 5 | Maximum number of addends per expression |
| `--max_value` | 100 | Maximum value of numbers |
| `--min_numbers` | 2 | Minimum number of addends per expression |
| `--output_dir` | `./data_output` | Output directory |
| `--train_ratio` | 0.8 | Training set ratio |
| `--val_ratio` | 0.1 | Validation set ratio |
| `--test_ratio` | 0.1 | Test set ratio |
| `--seed` | 42 | Random seed |

### Configuration Examples

#### Small-scale dataset (suitable for quick experiments)
```bash
python data/generate.py \
    --num_samples 1000 \
    --max_numbers 3 \
    --max_value 50
```

#### Large-scale dataset (suitable for full training)
```bash
python data/generate.py \
    --num_samples 500000 \
    --max_numbers 8 \
    --max_value 999
```

#### Simple dataset (suitable for debugging)
```bash
python data/generate.py \
    --num_samples 100 \
    --max_numbers 2 \
    --max_value 10
```

## Transformer Integration

### 1. Data Loader

```python
import json
import torch
from torch.utils.data import Dataset, DataLoader

class MathDataset(Dataset):
    def __init__(self, data_path, max_length=20):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Input sequence padding
        input_ids = item['input_ids'][:self.max_length]
        input_ids += [0] * (self.max_length - len(input_ids))
        
        # Output sequence padding
        output_ids = item['output_ids'][:self.max_length]
        output_ids += [0] * (self.max_length - len(output_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long)
        }

# Usage example
train_dataset = MathDataset('data_output/train.json')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### 2. Training Script Example

```python
import torch
import torch.nn as nn
from transformer import Transformer

# Load vocabulary
with open('data_output/vocab.json', 'r') as f:
    vocab_data = json.load(f)
    vocab_size = vocab_data['vocab_size']

# Create model
model = Transformer(
    vocab_size=vocab_size,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_len=50
)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token

# Training loop
model.train()
for epoch in range(100):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids']
        output_ids = batch['output_ids']
        
        # Forward pass
        outputs = model(input_ids)
        
        # Calculate loss
        loss = criterion(
            outputs.view(-1, vocab_size), 
            output_ids.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
```

### 3. Inference Example

```python
def predict_math(model, generator, expression):
    """Predict mathematical expression result"""
    model.eval()
    
    # Build input
    input_text = f"<START>{expression}="
    input_ids = generator.tokenize(input_text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    with torch.no_grad():
        # Generate output
        output_ids = []
        current_input = input_tensor
        
        for _ in range(10):  # Maximum generation length
            outputs = model(current_input)
            next_token = torch.argmax(outputs[0, -1, :])
            
            if next_token.item() == generator.vocab['<END>']:
                break
                
            output_ids.append(next_token.item())
            current_input = torch.cat([
                current_input, 
                next_token.unsqueeze(0).unsqueeze(0)
            ], dim=1)
        
        # Decode result
        result_text = generator.detokenize(output_ids)
        return result_text

# Usage example
expression = "25+37+14"
predicted_result = predict_math(model, generator, expression)
print(f"{expression} = {predicted_result}")
```

## Data Analysis

### 1. Dataset Statistics

The generator automatically provides detailed statistics:

```
=== Training Set Statistics ===
Dataset size: 8000
Input sequence length - Min: 5, Max: 15, Average: 9.14
Output sequence length - Min: 2, Max: 3, Average: 2.77

Sample data:
Example 1:
  Expression: 92+44+27
  Result: 163
  Input: <START>92+44+27=
  Output: 163<END>
```

### 2. Complexity Analysis

#### Sequence Length Distribution
- **Shortest input**: `<START>1+2=` (5 tokens)
- **Longest input**: `<START>99+98+97+96+95=` (15 tokens)
- **Average length**: About 9 tokens

#### Value Range
- **Minimum result**: 2 (1+1)
- **Maximum result**: 500 (5×100)
- **Result distribution**: Relatively uniform

### 3. Difficulty Levels

```python
# Simple level
--max_numbers 2 --max_value 10
# Example: 3+7=10

# Medium level
--max_numbers 4 --max_value 50  
# Example: 23+31+45+12=111

# Hard level
--max_numbers 6 --max_value 999
# Example: 234+567+890+123+456+789=3059
```

## Extended Features

### 1. Support for Other Operations

```python
class ExtendedMathGenerator(MathDataGenerator):
    def __init__(self, operations=['+'], **kwargs):
        super().__init__(**kwargs)
        self.operations = operations
        
    def generate_expression(self):
        # Extend support for subtraction, multiplication, etc.
        operation = random.choice(self.operations)
        if operation == '+':
            return super().generate_expression()
        elif operation == '-':
            # Implement subtraction logic
            pass
        elif operation == '*':
            # Implement multiplication logic
            pass
```

### 2. Data Augmentation

```python
def augment_dataset(dataset, factor=2):
    """Data augmentation: rearrange addend order"""
    augmented = []
    for item in dataset:
        numbers = item['expression'].split('+')
        for _ in range(factor):
            random.shuffle(numbers)
            new_expr = '+'.join(numbers)
            # Create new sample...
    return augmented
```

### 3. Validation Tools

```python
def validate_dataset(dataset):
    """Validate dataset correctness"""
    errors = 0
    for item in dataset:
        expression = item['expression']
        expected = item['result']
        actual = eval(expression)  # Only for validation
        
        if actual != expected:
            errors += 1
            print(f"Error: {expression} = {expected}, should be {actual}")
    
    print(f"Validation complete, errors: {errors}/{len(dataset)}")
```

### 4. Visualization Analysis

```python
import matplotlib.pyplot as plt

def plot_length_distribution(dataset):
    """Plot sequence length distribution"""
    lengths = [len(item['input_ids']) for item in dataset]
    plt.hist(lengths, bins=20)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Input Sequence Length Distribution')
    plt.show()

def plot_result_distribution(dataset):
    """Plot result distribution"""
    results = [item['result'] for item in dataset]
    plt.hist(results, bins=50)
    plt.xlabel('Calculation Result')
    plt.ylabel('Frequency')
    plt.title('Addition Result Distribution')
    plt.show()
```

## Best Practices

### 1. Dataset Size Recommendations

| Model Scale | Recommended Samples | Parameter Configuration |
|-------------|-------------------|------------------------|
| Small experiments | 1,000-10,000 | `--max_numbers 3 --max_value 50` |
| Medium training | 50,000-100,000 | `--max_numbers 5 --max_value 100` |
| Large training | 500,000+ | `--max_numbers 8 --max_value 999` |

### 2. Training Recommendations

- **Learning rate**: Start from 1e-4, adjust based on convergence
- **Batch size**: 32-128, adjust based on GPU memory
- **Sequence length**: Recommend setting to 1.2x the maximum dataset length
- **Early stopping**: Monitor validation loss to prevent overfitting

### 3. Evaluation Metrics

```python
def calculate_accuracy(model, test_loader):
    """Calculate exact match accuracy"""
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            predictions = model.generate(batch['input_ids'])
            targets = batch['output_ids']
            
            # Compare if sequences match exactly
            for pred, target in zip(predictions, targets):
                if torch.equal(pred, target):
                    correct += 1
                total += 1
    
    return correct / total
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce sample count
   - Lower maximum sequence length
   - Use smaller batch size

2. **Training Not Converging**
   - Check learning rate settings
   - Verify data format correctness
   - Increase model capacity

3. **Incorrect Generation Results**
   - Check vocabulary mapping
   - Verify tokenization logic
   - Confirm special token handling

### Debugging Tips

```python
# Print samples for inspection
for i, item in enumerate(dataset[:5]):
    print(f"Sample {i}:")
    print(f"  Original: {item['expression']} = {item['result']}")
    print(f"  Input: {item['input_text']}")
    print(f"  Output: {item['output_text']}")
    print(f"  Verification: {eval(item['expression'])} == {item['result']}")
    print()
```

## Performance Optimization

### 1. Memory Optimization
- Use gradient checkpointing for large models
- Implement dynamic batching based on sequence length
- Consider mixed precision training (FP16)

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
def validate_generator():
    """Validate data generator implementation"""
    generator = MathDataGenerator(max_numbers=3, max_value=20)
    
    # Test expression generation
    for _ in range(5):
        expr, result = generator.generate_expression()
        actual = eval(expr)
        assert actual == result, f"Mismatch: {expr} = {result}, actual = {actual}"
        print(f"✅ {expr} = {result}")
    
    # Test tokenization
    test_text = "<START>12+34="
    tokens = generator.tokenize(test_text)
    recovered = generator.detokenize(tokens)
    assert recovered == test_text, f"Tokenization error: {test_text} != {recovered}"
    print("✅ Tokenization test passed")
    
    # Test dataset generation
    dataset = generator.generate_dataset(10)
    assert len(dataset) == 10, "Dataset size mismatch"
    print("✅ Dataset generation test passed")

# Run validation
validate_generator()
```

## Integration Examples

### 1. Custom Training Loop

```python
def train_math_transformer():
    """Complete training example"""
    # Generate data
    generator = MathDataGenerator(max_numbers=4, max_value=100)
    dataset = generator.generate_dataset(10000)
    train_set, val_set, test_set = generator.split_dataset(dataset)
    
    # Create data loaders
    train_loader = DataLoader(MathDataset(train_set), batch_size=32, shuffle=True)
    val_loader = DataLoader(MathDataset(val_set), batch_size=32, shuffle=False)
    
    # Initialize model
    model = Transformer(vocab_size=generator.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training
    best_val_loss = float('inf')
    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Training step...
            pass
            
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # Validation step...
                pass
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
```

### 2. Hyperparameter Search

```python
def hyperparameter_search():
    """Search for optimal hyperparameters"""
    configs = [
        {'max_numbers': 3, 'max_value': 50, 'd_model': 128},
        {'max_numbers': 4, 'max_value': 100, 'd_model': 256},
        {'max_numbers': 5, 'max_value': 200, 'd_model': 512},
    ]
    
    results = []
    for config in configs:
        print(f"Testing config: {config}")
        
        # Generate data with this config
        generator = MathDataGenerator(
            max_numbers=config['max_numbers'],
            max_value=config['max_value']
        )
        
        # Train model with this config
        model = Transformer(
            vocab_size=generator.vocab_size,
            d_model=config['d_model']
        )
        
        # Train and evaluate...
        accuracy = train_and_evaluate(model, generator)
        results.append((config, accuracy))
    
    # Find best configuration
    best_config, best_accuracy = max(results, key=lambda x: x[1])
    print(f"Best config: {best_config}, accuracy: {best_accuracy}")
```

## Summary

This mathematical addition data generator provides an ideal learning environment for Transformer models:

- **High data quality**: All samples are validated
- **Standard format**: Fully compatible with sequence-to-sequence tasks
- **Scalability**: Easy to extend to other mathematical operations
- **Ease of use**: Simple command-line and programming interfaces

Through this tool, you can quickly generate high-quality training data to teach Transformer models basic mathematical computation abilities, laying the foundation for more complex reasoning tasks.
