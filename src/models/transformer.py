"""
Custom BERT and GPT-2 Architectures (Part II)
=============================================
These models are built entirely from scratch using the modern components:
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU Activation Function (Modern standard, replacing GELU)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import RMSNorm, RotaryEmbedding, GroupedQueryAttention


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU).
    
    Used in LLaMA, PaLM, and other modern models instead of standard GELU.
    Formula: Swish(xW_1) * (xW_2)
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        # We need two projections for the gating mechanism
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish is x * sigmoid(x), which is equivalent to F.silu(x)
        gate = F.silu(self.w1(x))
        x = gate * self.w2(x)
        return self.w3(x)


class CustomTransformerLayer(nn.Module):
    """A unified Transformer layer that can act as Encoder or Causal Decoder."""
    
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, 
                 mlp_ratio: int = 4, dropout: float = 0.1, is_causal: bool = False):
        super().__init__()
        self.is_causal = is_causal
        
        # Pre-normalization (modern standard vs post-norm in original Transformer)
        self.norm1 = RMSNorm(embed_dim)
        self.attn = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, dropout)
        
        self.norm2 = RMSNorm(embed_dim)
        
        # SwiGLU hidden dim is typically scaled differently, but roughly similar param count
        hidden_dim = int(2 * embed_dim * mlp_ratio / 3)
        # Ensure it's a multiple of 256 for optimal hardware usage
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.mlp = SwiGLU(embed_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, rope: RotaryEmbedding, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        
        # 1. Attention Block (with pre-norm and residual)
        h = self.norm1(x)
        
        if self.is_causal:
            # Create causal mask dynamically if not provided
            seq_len = x.size(1)
            if attention_mask is None:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                attention_mask = torch.zeros(seq_len, seq_len, device=x.device)
                attention_mask.masked_fill_(causal_mask, float('-inf'))
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        h = self.attn(h, attention_mask=attention_mask, rope=rope)
        x = x + h
        
        # 2. MLP Block (with pre-norm and residual)
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        
        return x


class CustomBERT(nn.Module):
    """Custom BERT (~110M parameters) using RoPE, GQA, and RMSNorm.
    
    Target Params:
    - hidden_size = 768
    - num_layers = 12
    - num_heads = 12
    - num_kv_heads = 4 (GQA: 3 Q-heads share 1 KV-head)
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 768, num_layers: int = 12,
                 num_heads: int = 12, num_kv_heads: int = 4, mlp_ratio: int = 4, 
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        
        # Token Embeddings (No Absolute Position Embeddings due to RoPE)
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(dropout)
        
        # RoPE applies directly inside the layers
        self.rope = RotaryEmbedding(embed_dim // num_heads)
        
        self.layers = nn.ModuleList([
            CustomTransformerLayer(embed_dim, num_heads, num_kv_heads, mlp_ratio, dropout, is_causal=False)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(embed_dim)
        
        # Masked Language Modeling Head
        self.mlm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Weight tying (share weights between embedding and MLM head)
        self.mlm_head.weight = self.token_emb.weight
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)
        
        # Create padding mask [batch, 1, 1, seq_len]
        mask = (input_ids == self.pad_id).float() * -1e9
        attention_mask = mask.unsqueeze(1).unsqueeze(2)
        
        for layer in self.layers:
            x = layer(x, self.rope, attention_mask=attention_mask)
            
        x = self.norm(x)
        logits = self.mlm_head(x)
        return logits
    
    def extract_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states (used for the fusion task)."""
        x = self.token_emb(input_ids)
        mask = (input_ids == self.pad_id).float() * -1e9
        attention_mask = mask.unsqueeze(1).unsqueeze(2)
        
        for layer in self.layers:
            x = layer(x, self.rope, attention_mask=attention_mask)
            
        return self.norm(x)


class CustomGPT2(nn.Module):
    """Custom GPT-2 (~124M parameters) using RoPE, GQA, and RMSNorm.
    
    Target Params:
    - hidden_size = 768
    - num_layers = 12
    - num_heads = 12
    - num_kv_heads = 4
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 768, num_layers: int = 12,
                 num_heads: int = 12, num_kv_heads: int = 4, mlp_ratio: int = 4, 
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(dropout)
        
        self.rope = RotaryEmbedding(embed_dim // num_heads)
        
        self.layers = nn.ModuleList([
            CustomTransformerLayer(embed_dim, num_heads, num_kv_heads, mlp_ratio, dropout, is_causal=True)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(embed_dim)
        
        # Causal Language Modeling Head
        self.clm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Weight tying
        self.clm_head.weight = self.token_emb.weight
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = self.token_emb(input_ids)
        x = self.emb_dropout(x)
        
        for layer in self.layers:
            x = layer(x, self.rope)  # Causal mask is handled inside the layer
            
        x = self.norm(x)
        logits = self.clm_head(x)
        return logits


if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Testing Custom BERT...")
    bert = CustomBERT(vocab_size=32000)
    print(f"BERT Parameters: {count_parameters(bert):,}")
    
    print("\nTesting Custom GPT-2...")
    gpt2 = CustomGPT2(vocab_size=32000)
    print(f"GPT-2 Parameters: {count_parameters(gpt2):,}")
    
    dummy_input = torch.randint(0, 32000, (2, 50))
    bert_out = bert(dummy_input)
    gpt2_out = gpt2(dummy_input)
    
    print(f"\nBERT output shape: {bert_out.shape}")
    print(f"GPT-2 output shape: {gpt2_out.shape}")
    print("Smoke test passed!")
