"""
Modern Transformer Components for Pretraining (Part II)
=====================================================
Contains the architectural improvements required by the assignment:
- RMSNorm (Root Mean Square Normalization)
- RoPE (Rotary Positional Embeddings)
- GQA (Grouped Query Attention)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Standard LayerNorm: (x - mean) / std * weight + bias
    RMSNorm: x / RMS(x) * weight
    
    RMSNorm drops the mean-centering operation, arguing that the success
    of LayerNorm comes primarily from scaling invariance, not translation
    invariance. This reduces computation time while maintaining performance.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (no bias)
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS: sqrt(mean(x^2))
        # We use reciprocal square root (rsqrt) directly for efficiency
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        
        # Scale
        return self.weight * x_normed


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE).
    
    Instead of adding absolute positional embeddings to the input, RoPE applies
    a rotation in the complex plane to the Query and Key vectors inside the
    attention mechanism. The angle of rotation is proportional to the absolute
    position, which naturally encodes relative distance between tokens.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Calculate inverse frequencies for each pair of dimensions
        # theta_i = 10000^(-2(i-1)/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin caches
        self._build_cache(max_seq_len)
        
    def _build_cache(self, seq_len: int):
        # Position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Outer product to get all position * frequency combinations
        # freqs: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate each frequency to match [dim] for Q and K
        # emb: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin with shape [1, 1, seq_len, dim] to broadcast with [batch, num_heads, seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """Apply RoPE to Query and Key tensors.
        Args:
            q, k: Tensors of shape [batch, num_heads, seq_len, head_dim]
        """
        seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            
        # Get cached cos and sin
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        
        # Apply rotation:
        # [x1, x2] rotated by theta -> [x1*cos(t) - x2*sin(t), x2*cos(t) + x1*sin(t)]
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA).
    
    Standard MHA: N query heads, N key heads, N value heads (Memory intensive)
    MQA: N query heads, 1 key head, 1 value head (Loss of quality)
    GQA: N query heads, G key/value heads (Best of both worlds)
    
    By sharing KV heads among multiple query heads, we drastically reduce
    the memory footprint of the KV cache during autoregressive generation.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        num_kv_heads: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.head_dim = embed_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        
        # Linear projections
        # Q has num_heads, K and V have num_kv_heads
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None, 
                rope: RotaryEmbedding = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, embed_dim]
            attention_mask: [batch, 1, seq_len, seq_len] (optional)
            rope: RotaryEmbedding layer (optional)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if provided
        if rope is not None:
            q, k = rope(q, k)
            
        # Repeat KV heads for each group to match Q heads
        # k, v: [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_heads, seq_len, head_dim]
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        # Compute attention scores: Q @ K^T / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        
        # Compute weighted values: probs @ V
        output = torch.matmul(probs, v)
        
        # Reshape back to [batch, seq_len, embed_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection and dropout
        return self.resid_dropout(self.out_proj(output))
