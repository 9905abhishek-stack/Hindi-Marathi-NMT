"""
LSTM Seq2Seq with Bahdanau Attention for Hindi-Marathi Translation
===================================================================
Architecture:
    Encoder: Bidirectional multi-layer LSTM
    Attention: Bahdanau (additive) attention
    Decoder: Unidirectional multi-layer LSTM with attention + input feeding

This is the classical NMT architecture from Bahdanau et al. (2015) +
Luong et al. (2015) input feeding approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention Mechanism.
    
    Computes: score(s_t, h_j) = v^T * tanh(W_s * s_t + W_h * h_j)
    
    This is a small feedforward network that learns to score how relevant
    each encoder hidden state h_j is for the current decoder state s_t.
    
    Unlike dot-product attention (s_t^T * h_j), additive attention can
    handle cases where encoder and decoder have different hidden sizes.
    """
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int = 256):
        super().__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor, 
                encoder_mask: torch.Tensor = None):
        """
        Args:
            decoder_state: [batch, decoder_dim] - current decoder hidden state
            encoder_outputs: [batch, src_len, encoder_dim] - all encoder hidden states
            encoder_mask: [batch, src_len] - True for padding positions (to ignore)
        
        Returns:
            context: [batch, encoder_dim] - weighted sum of encoder outputs
            attn_weights: [batch, src_len] - attention distribution
        """
        # Project encoder and decoder states to attention space
        # encoder_proj: [batch, src_len, attn_dim]
        encoder_proj = self.W_encoder(encoder_outputs)
        # decoder_proj: [batch, 1, attn_dim] (unsqueeze for broadcasting)
        decoder_proj = self.W_decoder(decoder_state).unsqueeze(1)
        
        # Compute alignment scores: [batch, src_len, 1] -> [batch, src_len]
        energy = self.v(torch.tanh(encoder_proj + decoder_proj)).squeeze(-1)
        
        # Mask padding positions with -inf so softmax gives them 0 weight
        if encoder_mask is not None:
            energy = energy.masked_fill(encoder_mask, float('-inf'))
        
        # Normalize to get attention distribution
        attn_weights = F.softmax(energy, dim=-1)  # [batch, src_len]
        
        # Compute context vector: weighted sum of encoder outputs
        # [batch, 1, src_len] @ [batch, src_len, enc_dim] -> [batch, 1, enc_dim]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attn_weights


class Encoder(nn.Module):
    """Bidirectional LSTM Encoder.
    
    Reads the source sentence and produces a sequence of hidden states.
    Bidirectional means we run two LSTMs (forward + backward) and concatenate
    their outputs, giving each position context from both past and future.
    
    Output: hidden states of shape [batch, src_len, 2 * hidden_dim]
    (2x because bidirectional concatenation)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Project bidirectional hidden state (2*hidden) down to hidden for decoder
        self.hidden_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cell_proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, src_ids: torch.Tensor, src_lengths: torch.Tensor):
        """
        Args:
            src_ids: [batch, src_len] - source token IDs
            src_lengths: [batch] - actual lengths (for packing)
        
        Returns:
            encoder_outputs: [batch, src_len, hidden*2] - all hidden states
            hidden: (h, c) - each [num_layers, batch, hidden] - for decoder init
        """
        # Embed: [batch, src_len, embed_dim]
        embedded = self.dropout(self.embedding(src_ids))
        
        # Pack padded sequences for efficient LSTM processing
        # This tells the LSTM to skip padding positions
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu().clamp(min=1), 
            batch_first=True, enforce_sorted=False
        )
        
        # Run bidirectional LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack: [batch, src_len, hidden*2]
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        
        # Combine bidirectional hidden states for decoder initialization
        # hidden shape: [num_layers*2, batch, hidden] -> [num_layers, batch, hidden]
        # We concatenate forward and backward for each layer, then project
        batch_size = hidden.size(1)
        
        # Reshape: [num_layers, 2, batch, hidden] -> [num_layers, batch, hidden*2]
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)  # [num_layers, batch, hidden*2]
        
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=-1)
        
        # Project to decoder dimension
        hidden = torch.tanh(self.hidden_proj(hidden))  # [num_layers, batch, hidden]
        cell = torch.tanh(self.cell_proj(cell))
        
        return encoder_outputs, (hidden, cell)


class Decoder(nn.Module):
    """LSTM Decoder with Bahdanau Attention and Input Feeding.
    
    Input feeding (Luong et al., 2015): The context vector from attention is
    concatenated with the input embedding at each step. This gives the decoder
    direct information about what parts of the source it attended to in the
    previous step, improving alignment quality.
    
    At each step:
    1. Concatenate [embedding(prev_token), prev_context] as input
    2. Run through LSTM
    3. Compute attention over encoder outputs
    4. Combine LSTM output and context for prediction
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        encoder_dim: int = 1024,  # hidden*2 because bidirectional encoder
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_dim: int = 256,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        
        # Input feeding: input = [embed_dim + encoder_dim]
        self.lstm = nn.LSTM(
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, attention_dim)
        
        # Combine LSTM output + context vector -> prediction
        self.output_proj = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.output_vocab = nn.Linear(hidden_dim, vocab_size)
    
    def forward_step(self, input_token, hidden, cell, encoder_outputs, 
                     encoder_mask, prev_context):
        """Single decoder step.
        
        Args:
            input_token: [batch] - previous token ID
            hidden: [num_layers, batch, hidden]
            cell: [num_layers, batch, hidden]
            encoder_outputs: [batch, src_len, encoder_dim]
            encoder_mask: [batch, src_len]
            prev_context: [batch, encoder_dim] - context from previous step
        
        Returns:
            logits: [batch, vocab_size]
            hidden, cell: updated states
            context: [batch, encoder_dim] - new context
            attn_weights: [batch, src_len]
        """
        # Embed input token: [batch, embed_dim]
        embedded = self.dropout(self.embedding(input_token))
        
        # Input feeding: concatenate embedding with previous context
        # [batch, embed_dim + encoder_dim]
        lstm_input = torch.cat([embedded, prev_context], dim=-1)
        
        # LSTM step: [batch, 1, input_dim] -> [batch, 1, hidden]
        lstm_output, (hidden, cell) = self.lstm(
            lstm_input.unsqueeze(1), (hidden, cell)
        )
        lstm_output = lstm_output.squeeze(1)  # [batch, hidden]
        
        # Compute attention
        context, attn_weights = self.attention(lstm_output, encoder_outputs, encoder_mask)
        
        # Combine LSTM output and context: [batch, hidden + encoder_dim]
        combined = torch.cat([lstm_output, context], dim=-1)
        output = self.dropout(torch.tanh(self.output_proj(combined)))
        
        # Project to vocabulary: [batch, vocab_size]
        logits = self.output_vocab(output)
        
        return logits, hidden, cell, context, attn_weights
    
    def forward(self, tgt_ids, encoder_outputs, encoder_hidden, encoder_mask,
                teacher_forcing_ratio=1.0):
        """Full decoder forward pass (training).
        
        Args:
            tgt_ids: [batch, tgt_len] - target token IDs (with BOS)
            encoder_outputs: [batch, src_len, encoder_dim]
            encoder_hidden: (h, c) from encoder
            encoder_mask: [batch, src_len]
            teacher_forcing_ratio: probability of using ground truth vs prediction
        
        Returns:
            logits: [batch, tgt_len-1, vocab_size] - predictions for each step
            attention_weights: [batch, tgt_len-1, src_len]
        """
        batch_size = tgt_ids.size(0)
        tgt_len = tgt_ids.size(1)
        encoder_dim = encoder_outputs.size(-1)
        
        hidden, cell = encoder_hidden
        
        # Initialize context vector with zeros
        prev_context = torch.zeros(batch_size, encoder_dim, device=tgt_ids.device)
        
        # Store outputs
        all_logits = []
        all_attn = []
        
        # First input is BOS token
        input_token = tgt_ids[:, 0]
        
        # Decode step by step (skip BOS, predict from position 1 onwards)
        for t in range(1, tgt_len):
            logits, hidden, cell, prev_context, attn = self.forward_step(
                input_token, hidden, cell, encoder_outputs, encoder_mask, prev_context
            )
            all_logits.append(logits)
            all_attn.append(attn)
            
            # Teacher forcing: use ground truth or own prediction
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt_ids[:, t]  # Ground truth
            else:
                input_token = logits.argmax(dim=-1)  # Own prediction
        
        # Stack: [batch, tgt_len-1, vocab_size]
        logits = torch.stack(all_logits, dim=1)
        attention_weights = torch.stack(all_attn, dim=1)
        
        return logits, attention_weights


class Seq2Seq(nn.Module):
    """Complete Seq2Seq model combining Encoder, Decoder, and Attention.
    
    This wraps everything into a clean interface for training and inference.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_dim: int = 256,
        padding_idx: int = 0,
    ):
        super().__init__()
        encoder_dim = hidden_dim * 2  # Bidirectional
        
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_dim=encoder_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_dim=attention_dim,
            padding_idx=padding_idx,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization for better gradient flow."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, src_ids, src_lengths, tgt_ids, teacher_forcing_ratio=1.0):
        """
        Args:
            src_ids: [batch, src_len]
            src_lengths: [batch]
            tgt_ids: [batch, tgt_len]
            teacher_forcing_ratio: float
        
        Returns:
            logits: [batch, tgt_len-1, vocab_size]
            attn_weights: [batch, tgt_len-1, src_len]
        """
        # Create encoder padding mask: True where padded
        encoder_mask = (src_ids == self.encoder.embedding.padding_idx)
        
        # Encode source
        encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
        
        # Decode target
        logits, attn_weights = self.decoder(
            tgt_ids, encoder_outputs, encoder_hidden, encoder_mask,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        
        return logits, attn_weights
    
    @torch.no_grad()
    def translate_greedy(self, src_ids, src_lengths, max_len=100, bos_id=2, eos_id=3):
        """Greedy decoding for inference.
        
        At each step, pick the token with highest probability.
        Simple but suboptimal - beam search is better.
        """
        self.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device
        encoder_dim = self.encoder.hidden_dim * 2
        
        encoder_mask = (src_ids == self.encoder.embedding.padding_idx)
        encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
        
        hidden, cell = encoder_hidden
        prev_context = torch.zeros(batch_size, encoder_dim, device=device)
        
        # Start with BOS token
        input_token = torch.full((batch_size,), bos_id, dtype=torch.long, device=device)
        
        generated = [input_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len):
            logits, hidden, cell, prev_context, _ = self.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs, encoder_mask, prev_context
            )
            
            next_token = logits.argmax(dim=-1)  # [batch]
            next_token = next_token.masked_fill(finished, 0)  # Pad finished sequences
            generated.append(next_token)
            
            finished = finished | (next_token == eos_id)
            if finished.all():
                break
            
            input_token = next_token
        
        return torch.stack(generated, dim=1)  # [batch, gen_len]
    
    @torch.no_grad()
    def translate_beam(self, src_ids, src_lengths, beam_width=5, max_len=100,
                       bos_id=2, eos_id=3, length_penalty=0.6):
        """Beam search decoding (single example at a time for simplicity).
        
        Maintains beam_width candidate translations and expands the most
        promising ones at each step.
        """
        self.eval()
        assert src_ids.size(0) == 1, "Beam search works on single examples"
        device = src_ids.device
        encoder_dim = self.encoder.hidden_dim * 2
        
        encoder_mask_orig = (src_ids == self.encoder.embedding.padding_idx)
        encoder_outputs, (h, c) = self.encoder(src_ids, src_lengths)
        
        # The encoder uses pack_padded_sequence which may produce outputs
        # shorter than src_ids. Trim or re-derive mask to match.
        enc_seq_len = encoder_outputs.size(1)
        encoder_mask = encoder_mask_orig[:, :enc_seq_len]
        
        # Expand for beam: [1, ...] -> [beam, ...]
        encoder_outputs = encoder_outputs.expand(beam_width, -1, -1).contiguous()
        encoder_mask = encoder_mask.expand(beam_width, -1).contiguous()
        h = h.expand(-1, beam_width, -1).contiguous()
        c = c.expand(-1, beam_width, -1).contiguous()
        prev_context = torch.zeros(beam_width, encoder_dim, device=device)
        
        # Initialize beams
        input_token = torch.full((beam_width,), bos_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_width, device=device)
        beam_sequences = [[bos_id] for _ in range(beam_width)]
        finished_beams = []
        
        for step in range(max_len):
            logits, h, c, prev_context, _ = self.decoder.forward_step(
                input_token, h, c, encoder_outputs, encoder_mask, prev_context
            )
            
            log_probs = F.log_softmax(logits, dim=-1)  # [beam, vocab]
            
            if step == 0:
                # First step: all beams are identical, only expand from beam 0
                scores = log_probs[0]  # [vocab]
                top_scores, top_ids = scores.topk(beam_width)
                beam_scores = top_scores
                for i in range(beam_width):
                    beam_sequences[i] = [bos_id, top_ids[i].item()]
                input_token = top_ids
            else:
                # Expand: [beam, vocab]
                next_scores = beam_scores.unsqueeze(1) + log_probs
                
                # Flatten and pick top-k
                flat_scores = next_scores.view(-1)
                top_scores, top_flat_ids = flat_scores.topk(beam_width * 2)
                
                beam_ids = top_flat_ids // self.decoder.vocab_size
                token_ids = top_flat_ids % self.decoder.vocab_size
                
                new_beams = []
                new_scores = []
                new_h = []
                new_c = []
                new_ctx = []
                
                for score, beam_id, token_id in zip(top_scores, beam_ids, token_ids):
                    bid = beam_id.item()
                    tid = token_id.item()
                    seq = beam_sequences[bid] + [tid]
                    
                    if tid == eos_id:
                        # Length-normalized score
                        lp = ((5 + len(seq)) / 6) ** length_penalty
                        finished_beams.append((score.item() / lp, seq))
                    else:
                        if len(new_beams) < beam_width:
                            new_beams.append(seq)
                            new_scores.append(score)
                            new_h.append(h[:, bid])
                            new_c.append(c[:, bid])
                            new_ctx.append(prev_context[bid])
                    
                    if len(new_beams) >= beam_width:
                        break
                
                if not new_beams:
                    break
                
                beam_sequences = new_beams
                beam_scores = torch.stack(new_scores)
                h = torch.stack(new_h, dim=1)
                c = torch.stack(new_c, dim=1)
                prev_context = torch.stack(new_ctx)
                input_token = torch.tensor(
                    [seq[-1] for seq in beam_sequences], dtype=torch.long, device=device
                )
                
                # Pad if we have fewer beams
                if len(beam_sequences) < beam_width:
                    pad_count = beam_width - len(beam_sequences)
                    beam_scores = F.pad(beam_scores, (0, pad_count), value=-1e9)
                    input_token = F.pad(input_token, (0, pad_count), value=bos_id)
                    h = F.pad(h, (0, 0, 0, pad_count))
                    c = F.pad(c, (0, 0, 0, pad_count))
                    prev_context = F.pad(prev_context, (0, 0, 0, pad_count))
        
        # Add unfinished beams
        for i, seq in enumerate(beam_sequences):
            lp = ((5 + len(seq)) / 6) ** length_penalty
            finished_beams.append((beam_scores[i].item() / lp, seq))
        
        # Return best beam
        finished_beams.sort(key=lambda x: x[0], reverse=True)
        best_seq = finished_beams[0][1] if finished_beams else [bos_id, eos_id]
        
        return torch.tensor([best_seq], dtype=torch.long, device=device)


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


if __name__ == "__main__":
    # Quick smoke test
    model = Seq2Seq(vocab_size=32000, embed_dim=512, hidden_dim=512, num_layers=2)
    params = count_parameters(model)
    print(f"Model parameters: {params:,}")
    
    # Dummy forward pass
    src = torch.randint(4, 100, (2, 10))
    tgt = torch.randint(4, 100, (2, 8))
    lengths = torch.tensor([10, 7])
    
    logits, attn = model(src, lengths, tgt)
    print(f"Logits shape: {logits.shape}")  # [2, 7, 32000]
    print(f"Attention shape: {attn.shape}")  # [2, 7, 10]
    print("Smoke test passed!")
