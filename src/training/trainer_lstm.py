"""
Training Pipeline for LSTM Seq2Seq Translation (Part I)
=======================================================
Handles:
- Training with teacher forcing + scheduled sampling
- Label smoothed cross-entropy loss
- Gradient clipping
- Mixed precision (fp16) for GPU efficiency
- Periodic validation with BLEU/CHRF++ metrics
- Checkpoint saving (best model by val BLEU)
- Metric logging for plot generation
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm

from src.models.lstm_seq2seq import Seq2Seq, count_parameters
from src.data.dataset import get_dataloaders
from src.data.tokenizer import Tokenizer
from src.evaluation.metrics import translate_and_evaluate
from src.training.checkpoint_utils import safe_save_checkpoint, safe_save_json, safe_load_checkpoint


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with label smoothing.
    
    Instead of hard targets [0, 0, 1, 0, ...], we use soft targets
    [eps/V, eps/V, 1-eps, eps/V, ...] where eps=0.1 and V=vocab_size.
    
    This prevents the model from being over-confident and acts as
    regularization, improving generalization for translation.
    """
    
    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch * seq_len, vocab_size]
            targets: [batch * seq_len]
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # NLL loss for the correct token
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Smooth loss: negative average of all log probs (uniform distribution)
        smooth_loss = -log_probs.sum(dim=-1) / self.vocab_size
        
        # Combine
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        # Mask padding
        mask = targets != self.padding_idx
        loss = (loss * mask).sum() / mask.sum()
        
        return loss


def get_teacher_forcing_ratio(epoch: int, total_epochs: int, 
                               start: float = 1.0, end: float = 0.5) -> float:
    """Linearly decay teacher forcing ratio.
    
    Start with full teacher forcing (stable training), gradually reduce
    so the model learns to recover from its own errors (scheduled sampling).
    """
    ratio = start - (start - end) * (epoch / max(total_epochs - 1, 1))
    return max(ratio, end)


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device,
                    teacher_forcing_ratio, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        src_ids = batch['src_ids'].to(device)
        src_lengths = batch['src_lengths'].to(device)
        tgt_ids = batch['tgt_ids'].to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda', dtype=torch.float16):
            logits, _ = model(src_ids, src_lengths, tgt_ids, teacher_forcing_ratio)
            
            # Target: shift right (predict tokens 1..T from inputs 0..T-1)
            target = tgt_ids[:, 1:]  # Remove BOS
            
            # Flatten for loss computation
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target.reshape(-1)
            
            loss = criterion(logits_flat, target_flat)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Track loss (count non-padding tokens)
        mask = target != 0
        num_tokens = mask.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def validate_loss(model, dataloader, criterion, device):
    """Compute validation loss."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for batch in dataloader:
        src_ids = batch['src_ids'].to(device)
        src_lengths = batch['src_lengths'].to(device)
        tgt_ids = batch['tgt_ids'].to(device)
        
        with autocast('cuda', dtype=torch.float16):
            logits, _ = model(src_ids, src_lengths, tgt_ids, teacher_forcing_ratio=1.0)
            target = tgt_ids[:, 1:]
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target.reshape(-1)
            loss = criterion(logits_flat, target_flat)
        
        mask = target != 0
        num_tokens = mask.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    
    return total_loss / max(total_tokens, 1)


def train(
    # Model config
    embed_dim: int = 512,
    hidden_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.3,
    attention_dim: int = 256,
    # Training config
    batch_size: int = 64,
    num_epochs: int = 30,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.1,
    tf_start: float = 1.0,
    tf_end: float = 0.5,
    # Data config
    data_dir: str = "data/processed",
    tokenizer_path: str = "data/tokenizer/bpe_32k.model",
    max_len: int = 128,
    direction: str = "hi2mr",
    # System config
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    experiment_name: str = "lstm_random",
    val_samples: int = 2000,
    eval_every: int = 1,
    resume: bool = False,
):
    """Full training pipeline."""
    
    checkpoint_dir = Path(checkpoint_dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Data ──
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Direction: {direction}")
    print(f"{'='*60}")
    
    print("\nLoading data...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        data_dir=data_dir, tokenizer_path=tokenizer_path,
        batch_size=batch_size, max_len=max_len, direction=direction,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # ── Model ──
    print("\nBuilding model...")
    model = Seq2Seq(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout,
        attention_dim=attention_dim, padding_idx=tokenizer.pad_id,
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"  Parameters: {num_params:,}")
    
    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    criterion = LabelSmoothingLoss(
        tokenizer.vocab_size, padding_idx=tokenizer.pad_id, smoothing=label_smoothing
    )
    scaler = GradScaler('cuda')
    
    # ── Metrics tracking ──
    history = {
        'train_loss': [], 'val_loss': [],
        'train_bleu': [], 'val_bleu': [],
        'train_chrf': [], 'val_chrf': [],
        'learning_rates': [], 'teacher_forcing': [],
    }
    best_val_bleu = 0
    start_epoch = 0
    
    # ── Resume logic (crash-safe) ──
    latest_ckpt_path = checkpoint_dir / 'latest_model.pt'
    history_path = checkpoint_dir / 'history.json'
    
    if resume:
        ckpt = safe_load_checkpoint(latest_ckpt_path, device)
        if ckpt is not None:
            print(f"\nResuming from checkpoint: {latest_ckpt_path}")
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_bleu = ckpt.get('best_val_bleu', 0)
            
            if history_path.exists():
                try:
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                except json.JSONDecodeError:
                    print("  [WARNING] history.json corrupted, starting fresh history.")
            print(f"  Resuming at epoch {start_epoch + 1}")
        else:
            print("\n  No valid checkpoint found. Starting from scratch.")
    
    # ── Training loop ──
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"  Device: {device}")
    print(f"  Mixed precision: fp16")
    print(f"  Label smoothing: {label_smoothing}")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Teacher forcing schedule
        tf_ratio = get_teacher_forcing_ratio(epoch, num_epochs, tf_start, tf_end)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e} | TF: {tf_ratio:.2f} ---")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, 
            device, tf_ratio, grad_clip
        )
        
        # Validation loss
        val_loss = validate_loss(model, val_loader, criterion, device)
        
        # Metrics (periodically - evaluation is slow)
        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            print("  Computing BLEU/CHRF++...")
            val_bleu, val_chrf, _, _ = translate_and_evaluate(
                model, val_loader, tokenizer, device, max_samples=val_samples
            )
            # Quick train metrics on subset
            train_bleu, train_chrf, _, _ = translate_and_evaluate(
                model, train_loader, tokenizer, device, max_samples=500
            )
        else:
            val_bleu = history['val_bleu'][-1] if history['val_bleu'] else 0
            val_chrf = history['val_chrf'][-1] if history['val_chrf'] else 0
            train_bleu = history['train_bleu'][-1] if history['train_bleu'] else 0
            train_chrf = history['train_chrf'][-1] if history['train_chrf'] else 0
        
        elapsed = time.time() - epoch_start
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_bleu'].append(train_bleu)
        history['val_bleu'].append(val_bleu)
        history['train_chrf'].append(train_chrf)
        history['val_chrf'].append(val_chrf)
        history['learning_rates'].append(current_lr)
        history['teacher_forcing'].append(tf_ratio)
        
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val BLEU-100: {val_bleu:.2f} | Val CHRF++-100: {val_chrf:.2f}")
        print(f"  Time: {elapsed:.0f}s")
        
        # LR scheduling based on val BLEU
        scheduler.step(val_bleu)
        
        # Save best model (atomic)
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            saved = safe_save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_bleu': val_bleu,
                'val_chrf': val_chrf,
                'best_val_bleu': best_val_bleu,
            }, checkpoint_dir / 'best_model.pt')
            if saved:
                print(f"  ** New best model saved! BLEU: {val_bleu:.2f} **")
            
        # Save latest model for resuming (atomic)
        safe_save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_bleu': best_val_bleu,
        }, checkpoint_dir / 'latest_model.pt')
        
        # Save history (atomic)
        safe_save_json(history, checkpoint_dir / 'history.json')
        print(f"  Epoch {epoch+1} checkpoint saved safely.")
    
    # ── Final evaluation on test set ──
    print(f"\n{'='*60}")
    print("Final evaluation on test set (beam search)...")
    print(f"{'='*60}")
    
    # Load best model
    ckpt = torch.load(checkpoint_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    test_bleu, test_chrf, hyps, refs = translate_and_evaluate(
        model, test_loader, tokenizer, device, use_beam=True, beam_width=5
    )
    
    print(f"\n  Test BLEU-100:   {test_bleu:.2f}")
    print(f"  Test CHRF++-100: {test_chrf:.2f}")
    
    # Save some example translations
    examples = []
    for i in range(min(20, len(hyps))):
        examples.append({'hypothesis': hyps[i], 'reference': refs[i]})
    
    with open(checkpoint_dir / 'test_examples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    # Final results
    results = {
        'experiment': experiment_name,
        'direction': direction,
        'test_bleu': test_bleu,
        'test_chrf': test_chrf,
        'best_val_bleu': best_val_bleu,
        'num_params': num_params,
        'num_epochs': num_epochs,
    }
    with open(checkpoint_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n[OK] Training complete!")
    return history, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='lstm_random', help='Experiment name')
    parser.add_argument('--direction', default='hi2mr', choices=['hi2mr', 'mr2hi'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=80)
    parser.add_argument('--eval_every', type=int, default=2)
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()
    
    train(
        experiment_name=args.experiment,
        direction=args.direction,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_len=args.max_len,
        eval_every=args.eval_every,
        resume=args.resume,
    )

