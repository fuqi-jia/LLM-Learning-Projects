"""
Transformer Implementation

This is a simplified Transformer implementation based on "Attention Is All You Need" (Vaswani et al., 2017).

Key differences from the original paper:
1. This is a DECODER-ONLY architecture (like GPT), not the full Encoder-Decoder architecture
2. Missing cross-attention mechanism (encoder-decoder attention)
3. Missing weight sharing between input/output embeddings
4. Simplified architecture suitable for sequence-to-sequence tasks like math problem solving

Original paper architecture:
- Full Encoder-Decoder with 6 layers each
- Encoder: Self-Attention + Feed-Forward
- Decoder: Self-Attention + Cross-Attention + Feed-Forward
- Both encoder and decoder use residual connections and layer normalization

This implementation:
- Stack of N identical transformer blocks (decoder-style)
- Each block: Self-Attention + Feed-Forward with residual connections
- Suitable for autoregressive generation tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism
    
    This implements the core attention mechanism from "Attention Is All You Need".
    The multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    
    Mathematical formulation:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    And Attention(Q, K, V) = softmax(QK^T / √d_k)V
    """
    
    def __init__(self, d_model, n_heads):
        """
        Initialize Multi-Head Attention
        
        Args:
            d_model (int): Model dimension (typically 512 in the original paper)
            n_heads (int): Number of attention heads (typically 8 in the original paper)
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension of each attention head
        
        # Ensure d_model is divisible by n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Linear projection layers for Query, Key, Value, and Output
        # In the original paper, these are the W^Q, W^K, W^V, and W^O matrices
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Output projection
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled Dot-Product Attention
        
        This is the core attention function: Attention(Q, K, V) = softmax(QK^T / √d_k)V
        
        Args:
            Q (Tensor): Query tensor [batch_size, n_heads, seq_len, d_k]
            K (Tensor): Key tensor [batch_size, n_heads, seq_len, d_k]
            V (Tensor): Value tensor [batch_size, n_heads, seq_len, d_k]
            mask (Tensor, optional): Attention mask to prevent attention to certain positions
            
        Returns:
            output (Tensor): Attention output [batch_size, n_heads, seq_len, d_k]
            attention_weights (Tensor): Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        # Compute attention scores: QK^T / √d_k
        # The scaling factor √d_k prevents the dot products from growing too large
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for causal attention or padding)
        if mask is not None:
            # Set masked positions to large negative value before softmax
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of Multi-Head Attention
        
        Args:
            query (Tensor): Query tensor [batch_size, seq_len, d_model]
            key (Tensor): Key tensor [batch_size, seq_len, d_model]
            value (Tensor): Value tensor [batch_size, seq_len, d_model]
            mask (Tensor, optional): Attention mask
            
        Returns:
            output (Tensor): Attention output [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Step 1: Linear transformations and split into multiple heads
        # Transform input through learned projection matrices
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)    # [batch_size, seq_len, d_model]
        V = self.w_v(value)  # [batch_size, seq_len, d_model]
        
        # Reshape and transpose for multi-head attention
        # Split d_model into n_heads * d_k and reorganize dimensions
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        
        # Step 2: Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Step 3: Concatenate heads and put through final linear layer
        # Transpose back and reshape to concatenate all heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )  # [batch_size, seq_len, d_model]
        
        # Final linear transformation (W^O in the paper)
        output = self.w_o(attention_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    This implements the feed-forward network applied to each position separately.
    From the paper: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    The original paper uses ReLU activation and has d_ff = 2048 when d_model = 512.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize Feed-Forward Network
        
        Args:
            d_model (int): Model dimension
            d_ff (int): Hidden dimension of feed-forward network (typically 4 * d_model)
            dropout (float): Dropout rate
        """
        super(FeedForward, self).__init__()
        # Two linear transformations with ReLU activation in between
        self.linear1 = nn.Linear(d_model, d_ff)    # First linear layer: d_model -> d_ff
        self.linear2 = nn.Linear(d_ff, d_model)    # Second linear layer: d_ff -> d_model
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        
        Args:
            x (Tensor): Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor: Output tensor [batch_size, seq_len, d_model]
        """
        # Apply first linear transformation, ReLU activation, dropout, then second linear transformation
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    """
    Layer Normalization
    
    Normalizes the inputs across the features dimension.
    LayerNorm(x) = γ * (x - μ) / σ + β
    
    Where μ is the mean and σ is the standard deviation computed across the last dimension.
    γ and β are learnable parameters.
    
    Note: The original paper uses Layer Normalization as described in 
    "Layer Normalization" (Ba et al., 2016)
    """
    
    def __init__(self, d_model, eps=1e-6):
        """
        Initialize Layer Normalization
        
        Args:
            d_model (int): Model dimension
            eps (float): Small value to prevent division by zero
        """
        super(LayerNorm, self).__init__()
        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(d_model))   # Scale parameter
        self.beta = nn.Parameter(torch.zeros(d_model))   # Shift parameter
        self.eps = eps
        
    def forward(self, x):
        """
        Apply layer normalization
        
        Args:
            x (Tensor): Input tensor [..., d_model]
            
        Returns:
            Tensor: Normalized tensor with same shape as input
        """
        # Compute mean and standard deviation along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Apply normalization: γ * (x - μ) / σ + β
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    
    Since the model contains no recurrence and no convolution, we add positional encodings
    to give the model some information about the relative or absolute position of tokens.
    
    The positional encodings use sine and cosine functions:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    This allows the model to easily learn to attend by relative positions.
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize Positional Encoding
        
        Args:
            d_model (int): Model dimension
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Create position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the division term for the sinusoidal pattern
        # div_term = 1 / (10000^(2i/d_model)) for i = 0, 1, 2, ..., d_model//2
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and transpose for easier indexing
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        # Register as buffer (not a parameter, but part of the model state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x (Tensor): Input embeddings [seq_len, batch_size, d_model] or [batch_size, seq_len, d_model]
            
        Returns:
            Tensor: Input embeddings + positional encoding
        """
        # Add positional encoding to input
        # x.size(0) should be seq_len if input format is [seq_len, batch_size, d_model]
        # For [batch_size, seq_len, d_model], we need x.size(1)
        if len(x.shape) == 3 and x.size(1) > x.size(0):  # Likely [batch_size, seq_len, d_model]
            return x + self.pe[:x.size(1), :].transpose(0, 1)
        else:  # [seq_len, batch_size, d_model]
            return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """
    Transformer Block (Decoder Layer)
    
    This implements one layer of the transformer decoder with:
    1. Multi-head self-attention with residual connection and layer normalization
    2. Position-wise feed-forward network with residual connection and layer normalization
    
    Note: This implementation uses POST-NORM (same as original paper):
    x = LayerNorm(x + Sublayer(x))
    
    Modern implementations often use PRE-NORM for better training stability:
    x = x + Sublayer(LayerNorm(x))
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Initialize Transformer Block
        
        Args:
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            d_ff (int): Feed-forward hidden dimension
            dropout (float): Dropout rate
        """
        super(TransformerBlock, self).__init__()
        
        # Sub-layer 1: Multi-head self-attention
        self.attention = MultiHeadAttention(d_model, n_heads)
        
        # Sub-layer 2: Position-wise feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization layers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass of transformer block
        
        Args:
            x (Tensor): Input tensor [batch_size, seq_len, d_model]
            mask (Tensor, optional): Attention mask
            
        Returns:
            Tensor: Output tensor [batch_size, seq_len, d_model]
        """
        # Sub-layer 1: Multi-head self-attention with residual connection and layer norm
        # In self-attention, query, key, and value are all the same (x)
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm
        
        # Sub-layer 2: Feed-forward network with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))    # Add & Norm
        
        return x


class Transformer(nn.Module):
    """
    Simplified Transformer Model (Decoder-Only Architecture)
    
    KEY DIFFERENCES FROM ORIGINAL PAPER:
    
    1. ARCHITECTURE TYPE:
       - Original: Full Encoder-Decoder architecture
       - This implementation: Decoder-only (like GPT)
       
    2. ATTENTION MECHANISMS:
       - Original Decoder: Self-attention + Cross-attention (encoder-decoder attention)
       - This implementation: Only self-attention
       
    3. USE CASE:
       - Original: Machine translation (encoder processes source, decoder generates target)
       - This implementation: Autoregressive generation (like language modeling)
       
    4. MISSING FEATURES:
       - No encoder stack
       - No cross-attention between encoder and decoder
       - No input/output embedding weight sharing
       - Simplified for educational purposes
    
    This architecture is more similar to GPT (decoder-only) than the original Transformer.
    It's suitable for tasks like:
    - Language modeling
    - Text generation
    - Sequence-to-sequence tasks where input and output share the same vocabulary
    """
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_len=5000, dropout=0.1):
        """
        Initialize Transformer Model
        
        Args:
            vocab_size (int): Size of the vocabulary
            d_model (int): Model dimension (512 in original paper)
            n_heads (int): Number of attention heads (8 in original paper)
            n_layers (int): Number of transformer blocks (6 in original paper)
            d_ff (int): Feed-forward hidden dimension (2048 in original paper)
            max_len (int): Maximum sequence length
            dropout (float): Dropout rate (0.1 in original paper)
        """
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Input Processing Layers
        
        # Token Embedding Layer
        # Converts token indices to dense vectors of dimension d_model
        # In the original paper, this is shared with the output projection weights
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        # Adds positional information to embeddings since transformer has no inherent
        # notion of sequence order (unlike RNNs or CNNs)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Core Transformer Architecture
        
        # Stack of N identical transformer blocks (decoder layers)
        # Each block contains self-attention and feed-forward sub-layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output Processing Layers
        
        # Final layer normalization (applied before output projection)
        self.layer_norm = LayerNorm(d_model)
        
        # Output projection layer (converts hidden states back to vocabulary logits)
        # In the original paper, this shares weights with the input embedding
        self.linear = nn.Linear(d_model, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights (not in original paper, but good practice)
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize model weights
        
        This is not specified in the original paper, but proper initialization
        is important for training stability.
        """
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Initialize linear layer weights
        nn.init.normal_(self.linear.weight, mean=0, std=0.02)
        
    def forward(self, x, mask=None):
        """
        Forward pass of the transformer
        
        Args:
            x (Tensor): Input token indices [batch_size, seq_len]
            mask (Tensor, optional): Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Tensor: Output logits [batch_size, seq_len, vocab_size]
        """
        # Step 1: Convert token indices to embeddings
        # The scaling by sqrt(d_model) is mentioned in the original paper
        # This prevents the positional encodings from dominating the word embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        
        # Step 2: Add positional encoding
        # This gives the model information about token positions
        x = self.positional_encoding(x)  # [batch_size, seq_len, d_model]
        
        # Step 3: Apply dropout to combined embeddings
        x = self.dropout(x)
        
        # Step 4: Pass through stack of transformer blocks
        # Each block applies self-attention and feed-forward transformations
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)  # [batch_size, seq_len, d_model]
            
        # Step 5: Apply final layer normalization
        x = self.layer_norm(x)  # [batch_size, seq_len, d_model]
        
        # Step 6: Project to vocabulary space
        # Convert hidden representations to vocabulary logits
        output = self.linear(x)  # [batch_size, seq_len, vocab_size]
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask for autoregressive generation
        
        This creates a mask that prevents the model from attending to future positions,
        which is essential for autoregressive generation (like language modeling).
        
        Args:
            sz (int): Sequence length
            
        Returns:
            Tensor: Causal mask [sz, sz]
        """
        # Create upper triangular matrix of ones
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.transpose(0, 1)
        
        # Convert to float and fill masked positions with -inf (will become 0 after softmax)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask
    
    def generate(self, start_tokens, max_length, temperature=1.0, top_k=None):
        """
        Generate sequences autoregressively
        
        This is a simple generation method for demonstration.
        More sophisticated methods include beam search, nucleus sampling, etc.
        
        Args:
            start_tokens (Tensor): Starting token sequence [batch_size, start_len]
            max_length (int): Maximum length to generate
            temperature (float): Sampling temperature (higher = more random)
            top_k (int, optional): Only consider top-k tokens for sampling
            
        Returns:
            Tensor: Generated sequences [batch_size, max_length]
        """
        self.eval()
        
        batch_size = start_tokens.size(0)
        current_tokens = start_tokens
        
        with torch.no_grad():
            for _ in range(max_length - start_tokens.size(1)):
                # Get model predictions
                logits = self.forward(current_tokens)  # [batch_size, seq_len, vocab_size]
                
                # Get logits for the last position
                next_token_logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    next_token_logits.fill_(float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
                
                # Append to sequence
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        return current_tokens


def create_padding_mask(seq, pad_idx=0):
    """
    Create padding mask to ignore padded positions
    
    Args:
        seq (Tensor): Input sequence [batch_size, seq_len]
        pad_idx (int): Padding token index
        
    Returns:
        Tensor: Padding mask [batch_size, 1, 1, seq_len]
    """
    # Create mask where True indicates real tokens, False indicates padding
    mask = (seq != pad_idx)
    
    # Reshape for attention computation: [batch_size, 1, 1, seq_len]
    # This allows broadcasting over attention heads and query positions
    return mask.unsqueeze(1).unsqueeze(2)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 50)
    print("TRANSFORMER MODEL ANALYSIS")
    print("=" * 50)
    
    # Model parameters (following original paper defaults)
    vocab_size = 10000
    d_model = 512      # Model dimension
    n_heads = 8        # Number of attention heads
    n_layers = 6       # Number of transformer layers
    d_ff = 2048        # Feed-forward dimension
    max_len = 100      # Maximum sequence length
    
    print(f"Model Configuration:")
    print(f"  - Vocabulary size: {vocab_size:,}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Attention heads: {n_heads}")
    print(f"  - Transformer layers: {n_layers}")
    print(f"  - Feed-forward dimension: {d_ff}")
    print(f"  - Maximum sequence length: {max_len}")
    
    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len
    )
    
    # Example input
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))  # Avoid padding token (0)
    
    print(f"\nInput Analysis:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Input shape: {input_ids.shape}")
    
    # Create padding mask
    padding_mask = create_padding_mask(input_ids)
    print(f"  - Padding mask shape: {padding_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids, padding_mask)
        
    print(f"\nOutput Analysis:")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Expected shape: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32
    
    print(f"\nArchitecture Comparison with Original Paper:")
    print(f"  ✅ Multi-Head Attention: Implemented")
    print(f"  ✅ Position-wise Feed-Forward: Implemented")
    print(f"  ✅ Positional Encoding: Implemented")
    print(f"  ✅ Residual Connections: Implemented")
    print(f"  ✅ Layer Normalization: Implemented")
    print(f"  ❌ Encoder Stack: Not implemented (decoder-only)")
    print(f"  ❌ Cross-Attention: Not implemented (decoder-only)")
    print(f"  ❌ Input/Output Embedding Sharing: Not implemented")
    
    print(f"\nThis is a DECODER-ONLY architecture similar to GPT,")
    print(f"not the full Encoder-Decoder architecture from the original paper.")
    print("=" * 50) 