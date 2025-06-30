#!/usr/bin/env python3
"""
Simple test script to verify training functionality
"""

import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add transformer directory to path
sys.path.append('./transformer')

try:
    from transformer import Transformer
    print("✅ Successfully imported Transformer")
except ImportError as e:
    print(f"❌ Failed to import Transformer: {e}")
    sys.exit(1)

class SimpleMathDataset(Dataset):
    """Simplified dataset for testing"""
    
    def __init__(self, data_path, max_len=30):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.max_len = max_len
        
        # Filter out sequences that are too long
        self.data = [
            item for item in self.data 
            if len(item['input_ids']) + len(item['output_ids']) <= max_len
        ]
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine input and output for autoregressive training
        # Format: input_ids + output_ids
        sequence = item['input_ids'] + item['output_ids']
        
        # Pad to max length
        if len(sequence) < self.max_len:
            sequence = sequence + [0] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]
        
        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'input_length': len(item['input_ids']),
            'expression': item['expression'],
            'result': item['result']
        }

def test_model_creation():
    """Test basic model creation"""
    print("\n🧪 Testing model creation...")
    
    # Load vocabulary
    with open('dataset/vocab.json', 'r') as f:
        vocab_data = json.load(f)
    
    vocab_size = vocab_data['vocab_size']
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=128,  # Smaller for testing
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_len=30,
        dropout=0.1
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, vocab_data

def test_data_loading():
    """Test data loading"""
    print("\n📊 Testing data loading...")
    
    # Create dataset
    dataset = SimpleMathDataset('dataset/train.json', max_len=30)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test one batch
    batch = next(iter(dataloader))
    print(f"✅ Batch shape: {batch['sequence'].shape}")
    print(f"✅ Sample expression: {batch['expression'][0]}")
    print(f"✅ Sample result: {batch['result'][0]}")
    
    return dataloader

def test_forward_pass(model, dataloader):
    """Test forward pass"""
    print("\n🚀 Testing forward pass...")
    
    model.eval()
    batch = next(iter(dataloader))
    sequence = batch['sequence']
    
    with torch.no_grad():
        # Forward pass
        outputs = model(sequence)
        print(f"✅ Output shape: {outputs.shape}")
        print(f"✅ Expected shape: [batch_size={sequence.size(0)}, seq_len={sequence.size(1)}, vocab_size={model.vocab_size}]")
    
    return outputs

def test_training_step(model, dataloader):
    """Test one training step"""
    print("\n🏋️ Testing training step...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    batch = next(iter(dataloader))
    sequence = batch['sequence']
    
    # Create input and target sequences for autoregressive training
    inputs = sequence[:, :-1]  # All but last token
    targets = sequence[:, 1:]  # All but first token
    
    # Forward pass
    optimizer.zero_grad()
    outputs = model(inputs)
    
    # Calculate loss
    loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"✅ Training step completed, Loss: {loss.item():.4f}")
    return loss.item()

def test_generation(model, vocab_data):
    """Test sequence generation"""
    print("\n🎯 Testing sequence generation...")
    
    model.eval()
    
    # Create a simple input: "<START>5+3="
    # Token IDs: <START>=1, 5=10, +=3, 3=8, ==4
    input_sequence = torch.tensor([[1, 10, 3, 8, 4]], dtype=torch.long)
    
    print(f"Input sequence: {input_sequence}")
    
    with torch.no_grad():
        # Generate a few tokens
        current_seq = input_sequence
        
        for i in range(5):  # Generate up to 5 tokens
            outputs = model(current_seq)
            next_token = torch.argmax(outputs[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
            print(f"Generated token {i+1}: {next_token.item()}")
            
            # Stop if we predict END token
            if next_token.item() == 2:  # <END> token
                current_seq = torch.cat([current_seq, next_token], dim=1)
                break
            
            current_seq = torch.cat([current_seq, next_token], dim=1)
    
    print(f"✅ Final sequence: {current_seq}")
    
    # Convert to text
    id_to_token = vocab_data['id_to_token']
    sequence_text = ''.join([id_to_token[str(x.item())] for x in current_seq[0]])
    print(f"✅ Sequence as text: {sequence_text}")

def main():
    print("=" * 50)
    print("🧪 TRANSFORMER TRAINING TEST")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists('dataset/train.json'):
        print("❌ Dataset not found. Please run ./generate.sh first")
        sys.exit(1)
    
    try:
        # Test model creation
        model, vocab_data = test_model_creation()
        
        # Test data loading
        dataloader = test_data_loading()
        
        # Test forward pass
        outputs = test_forward_pass(model, dataloader)
        
        # Test training step
        loss = test_training_step(model, dataloader)
        
        # Test generation
        test_generation(model, vocab_data)
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Model creation: OK")
        print("✅ Data loading: OK")
        print("✅ Forward pass: OK")
        print("✅ Training step: OK")
        print("✅ Generation: OK")
        print("\n🚀 Ready to start full training!")
        print("Run: python transformer/train_transformer.py")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 