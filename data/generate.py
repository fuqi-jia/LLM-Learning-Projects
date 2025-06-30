import random
import json
import os
from typing import List, Tuple, Dict
import argparse


class MathDataGenerator:
    """Math Data Generator"""
    
    def __init__(self, max_numbers=5, max_value=100, min_numbers=2, max_result=200):
        """
        Initialize the data generator
        
        Args:
            max_numbers: Maximum number of addends
            max_value: Maximum value of numbers
            min_numbers: Minimum number of addends
            max_result: Maximum result value for regularization
        """
        self.max_numbers = max_numbers
        self.max_value = max_value
        self.min_numbers = min_numbers
        self.max_result = max_result  # Add result limit for training stability
        
        # Build vocabulary
        self.build_vocab()
        
    def build_vocab(self):
        """Build vocabulary"""
        self.vocab = {}
        self.id_to_token = {}
        
        # Special tokens
        special_tokens = ['<PAD>', '<START>', '<END>', '+', '=']
        
        # Digit tokens (0-9)
        digit_tokens = [str(i) for i in range(10)]
        
        # Merge all tokens
        all_tokens = special_tokens + digit_tokens
        
        # Build vocabulary mapping
        for idx, token in enumerate(all_tokens):
            self.vocab[token] = idx
            self.id_to_token[idx] = token
            
        self.vocab_size = len(all_tokens)
        
    def generate_expression(self) -> Tuple[str, int]:
        """
        Generate an addition expression with regularization for training stability
        
        Returns:
            Tuple[str, int]: (expression, result)
        """
        # Randomly determine the number of addends
        num_count = random.randint(self.min_numbers, self.max_numbers)
        
        # Generate numbers with bias towards smaller values for training stability
        # Use exponential distribution to favor smaller numbers
        numbers = []
        for _ in range(num_count):
            # Generate with bias towards smaller numbers
            # 70% chance for numbers 1-20, 20% for 21-50, 10% for 51-max_value
            rand = random.random()
            if rand < 0.7:
                # Small numbers (1-20 or up to min(20, max_value))
                max_small = min(20, self.max_value)
                number = random.randint(1, max_small)
            elif rand < 0.9:
                # Medium numbers (21-50 or appropriate range)
                min_medium = min(21, self.max_value)
                max_medium = min(50, self.max_value)
                if min_medium <= max_medium:
                    number = random.randint(min_medium, max_medium)
                else:
                    number = random.randint(1, self.max_value)
            else:
                # Large numbers (51-max_value)
                min_large = min(51, self.max_value)
                if min_large <= self.max_value:
                    number = random.randint(min_large, self.max_value)
                else:
                    number = random.randint(1, self.max_value)
            
            numbers.append(number)
        
        # Check if result exceeds maximum allowed result
        result = sum(numbers)
        
        # If result is too large, regenerate with smaller numbers
        if result > self.max_result:
            # Fallback: generate smaller numbers that definitely fit
            target_avg = self.max_result // (num_count + 1)  # Leave some margin
            numbers = []
            for _ in range(num_count):
                # Generate numbers around the target average
                min_val = max(1, target_avg - 10)
                max_val = min(self.max_value, target_avg + 10)
                numbers.append(random.randint(min_val, max_val))
            
            result = sum(numbers)
            
            # Final safety check
            if result > self.max_result:
                # Last resort: use very small numbers
                numbers = [random.randint(1, min(10, self.max_value)) for _ in range(num_count)]
                result = sum(numbers)
        
        # Build expression
        expression = '+'.join(map(str, numbers))
        
        return expression, result
        
    def tokenize(self, text: str) -> List[int]:
        """
        Convert text to token id list
        
        Args:
            text: Input text
            
        Returns:
            List[int]: Token id list
        """
        tokens = []
        i = 0
        while i < len(text):
            if text[i].isdigit():
                # Process multiple digits
                num_str = ''
                while i < len(text) and text[i].isdigit():
                    num_str += text[i]
                    i += 1
                # Split multiple digits into single digits
                for digit in num_str:
                    tokens.append(self.vocab[digit])
            elif text[i] in self.vocab:
                tokens.append(self.vocab[text[i]])
                i += 1
            else:
                i += 1  # Skip unknown characters
        return tokens
        
    def detokenize(self, token_ids: List[int]) -> str:
        """
        Convert token id list to text
        
        Args:
            token_ids: Token id list
            
        Returns:
            str: Text
        """
        tokens = [self.id_to_token.get(id, '<UNK>') for id in token_ids]
        return ''.join(tokens)
        
    def create_training_example(self) -> Dict[str, List[int]]:
        """
        Create a training example
        
        Returns:
            Dict: Dictionary containing input and output
        """
        expression, result = self.generate_expression()
        
        # Create input sequence: <START> + expression + = 
        input_text = f"<START>{expression}="
        input_ids = self.tokenize(input_text)
        
        # Create output sequence: result + <END>
        output_text = f"{result}<END>"
        output_ids = self.tokenize(output_text)
        
        return {
            'input_ids': input_ids,
            'output_ids': output_ids,
            'expression': expression,
            'result': result,
            'input_text': input_text,
            'output_text': output_text
        }
        
    def generate_dataset(self, num_samples: int) -> List[Dict]:
        """
        Generate dataset
        
        Args:
            num_samples: Number of samples
            
        Returns:
            List[Dict]: Dataset
        """
        dataset = []
        seen_expressions = set()
        
        while len(dataset) < num_samples:
            example = self.create_training_example()
            
            # Avoid duplicate expressions
            if example['expression'] not in seen_expressions:
                seen_expressions.add(example['expression'])
                dataset.append(example)
                
        return dataset
        
    def split_dataset(self, dataset: List[Dict], 
                     train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, 
                     test_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset
        
        Args:
            dataset: Complete dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Tuple: (training set, validation set, test set)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "The sum of the ratios must be 1"
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_set = dataset[:train_size]
        val_set = dataset[train_size:train_size + val_size]
        test_set = dataset[train_size + val_size:]
        
        return train_set, val_set, test_set
        
    def save_dataset(self, dataset: List[Dict], filepath: str):
        """
        Save dataset to file
        
        Args:
            dataset: Dataset
            filepath: File path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
    def save_vocab(self, filepath: str):
        """
        Save vocabulary to file
        
        Args:
            filepath: File path
        """
        vocab_data = {
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'vocab_size': self.vocab_size
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            
    def load_vocab(self, filepath: str):
        """
        Load vocabulary from file
        
        Args:
            filepath: File path
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            
        self.vocab = vocab_data['vocab']
        self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        self.vocab_size = vocab_data['vocab_size']
        
    def print_stats(self, dataset: List[Dict]):
        """
        Print dataset statistics
        
        Args:
            dataset: Dataset
        """
        print(f"Dataset size: {len(dataset)}")
        
        input_lengths = [len(example['input_ids']) for example in dataset]
        output_lengths = [len(example['output_ids']) for example in dataset]
        results = [example['result'] for example in dataset]
        
        print(f"Input sequence length - Min: {min(input_lengths)}, Max: {max(input_lengths)}, Avg: {sum(input_lengths)/len(input_lengths):.2f}")
        print(f"Output sequence length - Min: {min(output_lengths)}, Max: {max(output_lengths)}, Avg: {sum(output_lengths)/len(output_lengths):.2f}")
        print(f"Result values - Min: {min(results)}, Max: {max(results)}, Avg: {sum(results)/len(results):.2f}")
        
        # Print distribution of result ranges for training stability analysis
        small_results = sum(1 for r in results if r <= 50)
        medium_results = sum(1 for r in results if 51 <= r <= 100)
        large_results = sum(1 for r in results if r > 100)
        
        print(f"Result distribution:")
        print(f"  Small (≤50): {small_results} ({small_results/len(results)*100:.1f}%)")
        print(f"  Medium (51-100): {medium_results} ({medium_results/len(results)*100:.1f}%)")
        print(f"  Large (>100): {large_results} ({large_results/len(results)*100:.1f}%)")
        
        # Print a few examples
        print("\nExample data:")
        for i, example in enumerate(dataset[:3]):
            print(f"Example {i+1}:")
            print(f"  Expression: {example['expression']}")
            print(f"  Result: {example['result']}")
            print(f"  Input: {example['input_text']}")
            print(f"  Output: {example['output_text']}")
            print(f"  Input IDs: {example['input_ids']}")
            print(f"  Output IDs: {example['output_ids']}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Math Addition Data Generator')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--max_numbers', type=int, default=5, help='Maximum number of addends')
    parser.add_argument('--max_value', type=int, default=100, help='Maximum value of numbers')
    parser.add_argument('--min_numbers', type=int, default=2, help='Minimum number of addends')
    parser.add_argument('--max_result', type=int, default=200, help='Maximum result value (for training stability)')
    parser.add_argument('--output_dir', type=str, default='./dataset', help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create data generator
    generator = MathDataGenerator(
        max_numbers=args.max_numbers,
        max_value=args.max_value,
        min_numbers=args.min_numbers,
        max_result=args.max_result
    )
    
    print("Start generating dataset...")
    print(f"Configuration: Number of samples={args.num_samples}, Addends={args.min_numbers}-{args.max_numbers}, Number range=1-{args.max_value}, Max result={args.max_result}")
    
    # Generate dataset
    dataset = generator.generate_dataset(args.num_samples)
    
    # Split dataset
    train_set, val_set, test_set = generator.split_dataset(
        dataset, args.train_ratio, args.val_ratio, args.test_ratio
    )
    
    # Save dataset
    generator.save_dataset(train_set, os.path.join(args.output_dir, 'train.json'))
    generator.save_dataset(val_set, os.path.join(args.output_dir, 'val.json'))
    generator.save_dataset(test_set, os.path.join(args.output_dir, 'test.json'))
    
    # Save vocabulary
    generator.save_vocab(os.path.join(args.output_dir, 'vocab.json'))
    
    # Print statistics
    print("\n=== Training set statistics ===")
    generator.print_stats(train_set)
    
    print("\n=== Validation set statistics ===")
    generator.print_stats(val_set)
    
    print("\n=== Test set statistics ===")
    generator.print_stats(test_set)
    
    print(f"\nDataset saved to: {args.output_dir}")
    print(f"Vocabulary size: {generator.vocab_size}")
    print("Generation completed!")


if __name__ == "__main__":
    main() 