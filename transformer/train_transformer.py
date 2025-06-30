"""
Complete Training Script for Math Addition Transformer

This script provides end-to-end training, validation, and testing functionality
for the Transformer model on math addition tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import argparse
import time
import math
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from transformer import Transformer

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class MathDataset(Dataset):
    """
    Dataset class for math addition data
    """
    
    def __init__(self, data_path: str, max_input_len: int = 20, max_output_len: int = 10):
        """
        Initialize dataset
        
        Args:
            data_path: Path to JSON data file
            max_input_len: Maximum input sequence length
            max_output_len: Maximum output sequence length
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        
        # Filter out sequences that are too long
        self.data = [
            item for item in self.data 
            if len(item['input_ids']) <= max_input_len and len(item['output_ids']) <= max_output_len
        ]
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Pad input sequence
        input_ids = item['input_ids'][:]
        input_length = len(input_ids)
        
        # Pad to max length
        input_ids += [0] * (self.max_input_len - len(input_ids))
        input_ids = input_ids[:self.max_input_len]
        
        # Pad output sequence
        output_ids = item['output_ids'][:]
        output_length = len(output_ids)
        
        # Pad to max length
        output_ids += [0] * (self.max_output_len - len(output_ids))
        output_ids = output_ids[:self.max_output_len]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long),
            'input_length': input_length,
            'output_length': output_length,
            'expression': item['expression'],
            'result': item['result']
        }


class MathTrainer:
    """
    Trainer class for math addition transformer
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader, vocab_data, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.vocab_data = vocab_data
        self.args = args
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            output_ids = batch['output_ids'].to(self.device)
            
            # Create combined sequence: input + output
            # For autoregressive training, we need to shift targets
            combined_seq = torch.cat([input_ids, output_ids], dim=1)
            
            # Input is everything except the last token
            inputs = combined_seq[:, :-1]
            # Targets are everything except the first token
            targets = combined_seq[:, 1:]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
                         # Calculate loss
             loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
             
             # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % self.args.log_interval == 0:
                print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def evaluate(self, data_loader, split_name="validation"):
        """Evaluate model on given dataset"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                output_ids = batch['output_ids'].to(self.device)
                
                # Create combined sequence
                combined_seq = torch.cat([input_ids, output_ids], dim=1)
                inputs = combined_seq[:, :-1]
                targets = combined_seq[:, 1:]
                
                # Forward pass
                outputs = self.model(inputs)
                
                                 # Calculate loss
                 loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
                 total_loss += loss.item()
                
                # Calculate accuracy (exact match for complete sequences)
                batch_correct = self.calculate_sequence_accuracy(batch, input_ids)
                correct_predictions += batch_correct
                total_predictions += input_ids.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions
        
        print(f"{split_name} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def calculate_sequence_accuracy(self, batch, input_ids):
        """Calculate exact sequence match accuracy"""
        correct = 0
        
        for i in range(input_ids.size(0)):
            # Generate prediction for this input
            input_seq = input_ids[i:i+1]  # Keep batch dimension
            predicted_output = self.generate_output(input_seq)
            true_output = batch['output_ids'][i]
            
            # Remove padding and compare
            true_output_clean = [x.item() for x in true_output if x.item() != 0]
            predicted_output_clean = [x for x in predicted_output if x != 0]
            
            if predicted_output_clean == true_output_clean:
                correct += 1
        
        return correct
    
    def generate_output(self, input_ids, max_output_len=10):
        """Generate output sequence for given input"""
        self.model.eval()
        
        with torch.no_grad():
            # Start with input sequence
            current_seq = input_ids.clone()
            
            for _ in range(max_output_len):
                # Get predictions
                outputs = self.model(current_seq)
                
                # Get next token (last position)
                next_token = torch.argmax(outputs[0, -1, :]).unsqueeze(0).unsqueeze(0)
                
                # Stop if we predict END token
                if next_token.item() == 2:  # <END> token
                    current_seq = torch.cat([current_seq, next_token], dim=1)
                    break
                
                # Add predicted token to sequence
                current_seq = torch.cat([current_seq, next_token], dim=1)
                
                # Stop if sequence gets too long
                if current_seq.size(1) > input_ids.size(1) + max_output_len:
                    break
        
        # Extract only the generated part (after input)
        generated_part = current_seq[0, input_ids.size(1):].tolist()
        return generated_part
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Training for {self.args.epochs} epochs")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print("-" * 50)
            
            # Training
            start_time = time.time()
            train_loss = self.train_epoch()
            train_time = time.time() - start_time
            
            print(f"Training Loss: {train_loss:.4f}, Time: {train_time:.2f}s")
            
            # Validation
            val_loss, val_accuracy = self.evaluate(self.val_loader, "Validation")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save training history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_val_loss = val_loss
                self.save_model('best_model.pth')
                print(f"New best model saved! Validation Accuracy: {val_accuracy:.4f}")
            
            # Early stopping
            if self.args.early_stopping and epoch > 10:
                recent_losses = self.val_losses[-5:]
                if len(recent_losses) == 5 and all(
                    recent_losses[i] >= recent_losses[i-1] for i in range(1, 5)
                ):
                    print("Early stopping triggered!")
                    break
        
        # Load best model for final evaluation
        self.load_model('best_model.pth')
        
        # Final evaluation on test set
        print("\n" + "="*50)
        print("FINAL EVALUATION")
        print("="*50)
        
        test_loss, test_accuracy = self.evaluate(self.test_loader, "Test")
        
        print(f"\nFinal Results:")
        print(f"Best Validation Accuracy: {self.best_val_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Show some example predictions
        self.show_examples()
        
        # Plot training curves
        self.plot_training_curves()
    
    def show_examples(self, num_examples=10):
        """Show example predictions"""
        print(f"\nExample Predictions:")
        print("-" * 60)
        
        self.model.eval()
        
        # Get a few examples from test set
        test_examples = []
        for i, batch in enumerate(self.test_loader):
            if i >= num_examples:
                break
            test_examples.append(batch)
        
        id_to_token = self.vocab_data['id_to_token']
        
        for i, batch in enumerate(test_examples):
            if i >= num_examples:
                break
                
            input_ids = batch['input_ids'][0:1].to(self.device)  # Take first example
            true_output = batch['output_ids'][0].tolist()
            expression = batch['expression'][0]
            true_result = batch['result'][0]
            
            # Generate prediction
            predicted_output = self.generate_output(input_ids)
            
            # Convert to text
            true_output_text = ''.join([id_to_token[str(x)] for x in true_output if x != 0])
            predicted_output_text = ''.join([id_to_token[str(x)] for x in predicted_output if x != 0])
            
            # Check if correct
            correct = "✓" if predicted_output == [x for x in true_output if x != 0] else "✗"
            
            print(f"Example {i+1}: {correct}")
            print(f"  Expression: {expression}")
            print(f"  True result: {true_result}")
            print(f"  True output: {true_output_text}")
            print(f"  Predicted: {predicted_output_text}")
            print()
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Training curves saved as 'training_curves.png'")
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy,
            'vocab_data': self.vocab_data,
            'args': self.args
        }, filepath)
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train Transformer for Math Addition')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Directory containing train.json, val.json, test.json, vocab.json')
    parser.add_argument('--max_input_len', type=int, default=20,
                        help='Maximum input sequence length')
    parser.add_argument('--max_output_len', type=int, default=10,
                        help='Maximum output sequence length')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log interval during training')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save models')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 50)
    print("MATH ADDITION TRANSFORMER TRAINING")
    print("=" * 50)
    print(f"Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    # Load vocabulary
    vocab_path = os.path.join(args.data_dir, 'vocab.json')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    vocab_size = vocab_data['vocab_size']
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    train_dataset = MathDataset(
        os.path.join(args.data_dir, 'train.json'),
        args.max_input_len, args.max_output_len
    )
    val_dataset = MathDataset(
        os.path.join(args.data_dir, 'val.json'),
        args.max_input_len, args.max_output_len
    )
    test_dataset = MathDataset(
        os.path.join(args.data_dir, 'test.json'),
        args.max_input_len, args.max_output_len
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()
    
    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_input_len + args.max_output_len,
        dropout=args.dropout
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer and start training
    trainer = MathTrainer(model, train_loader, val_loader, test_loader, vocab_data, args)
    trainer.train()


if __name__ == "__main__":
    main() 