"""
Trainer for Encoder-Decoder Fusion (Part II)
=============================================
Fine-tunes the combined BERT encoder + GPT-2 decoder model
for Hindi-Marathi translation.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.data.tokenizer import Tokenizer
from src.data.dataset import get_dataloaders
from src.models.fusion import TranslationFusionModel
from src.training.checkpoint_utils import safe_save_checkpoint, safe_load_checkpoint

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Tokenizer and Data
    print("Preparing Data...")
    tokenizer = Tokenizer(args.tokenizer_path)
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        direction=args.direction
    )
    
    # 2. Setup Model
    print("Initializing Fusion Model...")
    model = TranslationFusionModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_ratio=4,
        dropout=args.dropout,
        pad_id=tokenizer.pad_id
    ).to(device)
    
    # Load pretrained weights
    if args.bert_ckpt and os.path.exists(args.bert_ckpt):
        model.load_pretrained_encoder(args.bert_ckpt, device)
    else:
        print(f"Warning: BERT checkpoint not found at {args.bert_ckpt}. Encoder will be random.")
        
    if args.gpt2_ckpt and os.path.exists(args.gpt2_ckpt):
        model.load_pretrained_decoder(args.gpt2_ckpt, device)
    else:
        print(f"Warning: GPT-2 checkpoint not found at {args.gpt2_ckpt}. Decoder self-attn will be random.")
    
    # 3. Setup Optimizer and Loss
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    scaler = GradScaler()
    
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': []}
    
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / 'latest_model.pt'
    best_ckpt = checkpoint_dir / 'best_model.pt'
    
    if args.resume:
        ckpt = safe_load_checkpoint(latest_ckpt, device)
        if ckpt is not None:
            print(f"Resuming from {latest_ckpt}")
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            history = ckpt.get('history', {'train_loss': [], 'val_loss': []})
    
    best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
    
    # 4. Training Loop
    print(f"\nStarting Fusion Fine-tuning for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        
        # Stage-wise freezing: freeze encoder for first 2 epochs
        if epoch < 2:
            model.freeze_encoder()
        else:
            if epoch == 2:
                print("Epoch 2 reached: Unfreezing encoder for full end-to-end training.")
            model.unfreeze_encoder()
            
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in train_pbar:
            src = batch['src_ids'].to(device)
            tgt = batch['tgt_ids'].to(device)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            
            optimizer.zero_grad()
            
            with autocast():
                logits = model(src, tgt_input)
                loss = criterion(logits.reshape(-1, tokenizer.vocab_size), tgt_target.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in val_pbar:
                src = batch['src_ids'].to(device)
                tgt = batch['tgt_ids'].to(device)
                tgt_input = tgt[:, :-1]
                tgt_target = tgt[:, 1:]
                
                with autocast():
                    logits = model(src, tgt_input)
                    loss = criterion(logits.reshape(-1, tokenizer.vocab_size), tgt_target.reshape(-1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1} Summary: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'history': history,
        }
        
        # Save latest
        safe_save_checkpoint(state, latest_ckpt)
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            safe_save_checkpoint(state, best_ckpt)
            print("  New best validation loss! Saved best_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='fusion_hi2mr')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--tokenizer_path', type=str, default='data/tokenizer/bpe_32k.model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--direction', type=str, default='hi2mr')
    
    # Pretrained weights
    parser.add_argument('--bert_ckpt', type=str, default='checkpoints/bert_pretrain/latest_model.pt')
    parser.add_argument('--gpt2_ckpt', type=str, default='checkpoints/gpt2_pretrain/latest_model.pt')
    
    # Model config (must match pretraining)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=6) # 6 decoder layers for fusion
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_kv_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training config
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4) # Higher LR because frozen encoder initially
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    train(args)
