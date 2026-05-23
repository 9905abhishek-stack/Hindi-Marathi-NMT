"""
Encoder-Decoder Fusion for Translation (Part II)
==================================================
Combines pretrained BERT (encoder) + pretrained GPT-2 (decoder)
with Cross-Attention layers for Hindi-Marathi translation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import RMSNorm, RotaryEmbedding, GroupedQueryAttention
from src.models.transformer import CustomBERT, CustomGPT2, SwiGLU


class CrossAttention(nn.Module):
    """Cross-Attention for Encoder-Decoder fusion.
    
    Query comes from the decoder, Key/Value come from the encoder.
    Uses the same GQA-style head structure for efficiency.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        
        # Q from decoder, K/V from encoder
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, decoder_hidden: torch.Tensor, encoder_hidden: torch.Tensor,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            decoder_hidden: [batch, tgt_len, embed_dim]
            encoder_hidden: [batch, src_len, embed_dim]
            encoder_mask: [batch, 1, 1, src_len]
        """
        batch_size = decoder_hidden.size(0)
        tgt_len = decoder_hidden.size(1)
        src_len = encoder_hidden.size(1)
        
        q = self.q_proj(decoder_hidden).view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_hidden).view(batch_size, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_hidden).view(batch_size, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat KV heads
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if encoder_mask is not None:
            scores = scores + encoder_mask
        
        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        
        return self.resid_dropout(self.out_proj(output))


class FusionDecoderLayer(nn.Module):
    """Decoder layer with self-attention + cross-attention + FFN.
    
    This extends a standard GPT-2 layer by inserting a cross-attention
    block between the self-attention and feed-forward blocks.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int,
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # 1. Masked Self-Attention (from GPT-2)
        self.norm1 = RMSNorm(embed_dim)
        self.self_attn = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, dropout)
        
        # 2. Cross-Attention (NEW - fusion with encoder)
        self.norm_cross = RMSNorm(embed_dim)
        self.cross_attn = CrossAttention(embed_dim, num_heads, num_kv_heads, dropout)
        
        # 3. Feed-Forward
        self.norm2 = RMSNorm(embed_dim)
        hidden_dim = int(2 * embed_dim * mlp_ratio / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.mlp = SwiGLU(embed_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                rope: RotaryEmbedding, causal_mask: torch.Tensor = None,
                encoder_mask: torch.Tensor = None) -> torch.Tensor:
        
        # Self-Attention with causal masking
        h = self.norm1(x)
        h = self.self_attn(h, attention_mask=causal_mask, rope=rope)
        x = x + h
        
        # Cross-Attention with encoder
        h = self.norm_cross(x)
        h = self.cross_attn(h, encoder_output, encoder_mask)
        x = x + h
        
        # FFN
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        
        return x


class TranslationFusionModel(nn.Module):
    """Complete Encoder-Decoder model for Translation.
    
    Loads pretrained BERT as encoder, builds a new decoder with
    cross-attention layers, optionally initializing self-attention
    and FFN weights from pretrained GPT-2.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 768, num_layers: int = 6,
                 num_heads: int = 12, num_kv_heads: int = 4, mlp_ratio: int = 4,
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embed_dim = embed_dim
        
        # Encoder: pretrained BERT (frozen initially, optionally fine-tuned later)
        self.encoder = CustomBERT(vocab_size, embed_dim, num_layers=12,
                                  num_heads=num_heads, num_kv_heads=num_kv_heads,
                                  dropout=dropout, pad_id=pad_id)
        
        # Decoder: new layers with cross-attention
        self.decoder_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.decoder_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(embed_dim // num_heads)
        
        self.decoder_layers = nn.ModuleList([
            FusionDecoderLayer(embed_dim, num_heads, num_kv_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_norm = RMSNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        # Weight tying with decoder embedding
        self.output_proj.weight = self.decoder_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def load_pretrained_encoder(self, bert_checkpoint_path: str, device: str = 'cuda'):
        """Load pretrained BERT weights into the encoder."""
        ckpt = torch.load(bert_checkpoint_path, map_location=device, weights_only=False)
        self.encoder.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded pretrained BERT from {bert_checkpoint_path}")
    
    def load_pretrained_decoder(self, gpt2_checkpoint_path: str, device: str = 'cuda'):
        """Load pretrained GPT-2 weights for decoder self-attention and FFN.
        
        The cross-attention layers are randomly initialized (they didn't exist 
        in the original GPT-2) and will be learned during fine-tuning.
        """
        ckpt = torch.load(gpt2_checkpoint_path, map_location=device, weights_only=False)
        gpt2_state = ckpt['model_state_dict']
        
        # Copy token embedding
        if 'token_emb.weight' in gpt2_state:
            self.decoder_emb.weight.data.copy_(gpt2_state['token_emb.weight'])
        
        # Copy matching weights from GPT-2 layers to fusion decoder layers
        loaded = 0
        for i, layer in enumerate(self.decoder_layers):
            gpt2_layer_idx = i  # Map 1:1 (use first N GPT-2 layers)
            prefix = f'layers.{gpt2_layer_idx}.'
            
            # Copy self-attention weights
            for name in ['norm1.weight', 'attn.q_proj.weight', 'attn.k_proj.weight',
                         'attn.v_proj.weight', 'attn.out_proj.weight']:
                src_key = prefix + name
                if src_key in gpt2_state:
                    target_name = name.replace('attn.', 'self_attn.')
                    target_param = dict(layer.named_parameters()).get(target_name)
                    if target_param is not None and target_param.shape == gpt2_state[src_key].shape:
                        target_param.data.copy_(gpt2_state[src_key])
                        loaded += 1
            
            # Copy FFN weights
            for name in ['norm2.weight', 'mlp.w1.weight', 'mlp.w2.weight', 'mlp.w3.weight']:
                src_key = prefix + name
                if src_key in gpt2_state:
                    target_param = dict(layer.named_parameters()).get(name)
                    if target_param is not None and target_param.shape == gpt2_state[src_key].shape:
                        target_param.data.copy_(gpt2_state[src_key])
                        loaded += 1
        
        print(f"Loaded {loaded} parameter tensors from pretrained GPT-2")
    
    def freeze_encoder(self):
        """Freeze encoder weights (common for early fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder weights frozen.")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder weights unfrozen.")
    
    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor):
        """
        Args:
            src_ids: [batch, src_len] - source tokens (Hindi)
            tgt_ids: [batch, tgt_len] - target tokens (Marathi, with BOS)
        Returns:
            logits: [batch, tgt_len, vocab_size]
        """
        # Encode source
        encoder_output = self.encoder.extract_features(src_ids)  # [batch, src_len, embed_dim]
        
        # Encoder padding mask
        encoder_mask = (src_ids == self.pad_id).float() * -1e9  # [batch, src_len]
        encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2)   # [batch, 1, 1, src_len]
        
        # Decode target
        tgt_len = tgt_ids.size(1)
        x = self.decoder_emb(tgt_ids)
        x = self.decoder_dropout(x)
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt_ids.device), diagonal=1).bool()
        causal_mask = torch.zeros(tgt_len, tgt_len, device=tgt_ids.device).masked_fill(causal_mask, float('-inf'))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, self.rope, causal_mask, encoder_mask)
        
        x = self.decoder_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    @torch.no_grad()
    def translate_greedy(self, src_ids: torch.Tensor, src_lengths=None, max_len: int = 100,
                         bos_id: int = 2, eos_id: int = 3):
        """Greedy decoding for inference."""
        self.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device
        
        encoder_output = self.encoder.extract_features(src_ids)
        encoder_mask = (src_ids == self.pad_id).float() * -1e9
        encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
        
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len):
            logits = self._decode_step(generated, encoder_output, encoder_mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            next_token = next_token.masked_fill(finished.unsqueeze(1), 0)
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break
        
        return generated
    
    def _decode_step(self, tgt_ids, encoder_output, encoder_mask):
        """Helper for autoregressive decoding."""
        tgt_len = tgt_ids.size(1)
        x = self.decoder_emb(tgt_ids)
        
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt_ids.device), diagonal=1).bool()
        causal_mask = torch.zeros(tgt_len, tgt_len, device=tgt_ids.device).masked_fill(causal_mask, float('-inf'))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, self.rope, causal_mask, encoder_mask)
        
        x = self.decoder_norm(x)
        return self.output_proj(x)


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("Testing Fusion Model...")
    model = TranslationFusionModel(vocab_size=32000)
    
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    src = torch.randint(4, 100, (2, 20))
    tgt = torch.randint(4, 100, (2, 15))
    
    logits = model(src, tgt)
    print(f"Output shape: {logits.shape}")
    
    # Test greedy decoding
    gen = model.translate_greedy(src)
    print(f"Generated shape: {gen.shape}")
    
    print("Smoke test passed!")
