"""
GPT-2 Pretraining Pipeline (Causal Language Modeling)
=====================================================
Trains the custom GPT-2 decoder on the Marathi corpus
using Causal Language Modeling (CLM) / Autoregressive Training.
"""

import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

from src.models.transformer import CustomGPT2
from src.data.tokenizer import Tokenizer
from src.training.checkpoint_utils import safe_save_checkpoint, safe_load_checkpoint


class CLMDataset(Dataset):
    """Dataset for Causal Language Modeling.
    
    Reads text, tokenizes, and chunks.
    For GPT-2 (Decoder), we primarily pretrain on the target language (Marathi)
    to build a strong language model prior for translation generation.
    """
    def __init__(self, file_paths: list[str], tokenizer: Tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        
        print(f"Loading CLM data from {file_paths}...")
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Add EOS token so it learns sentence boundaries
                    tokens = self.tokenizer.encode(line) + [self.tokenizer.eos_id]
                    for i in range(0, len(tokens), max_len):
                        chunk = tokens[i:i + max_len]
                        if len(chunk) >= 5:
                            self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class DataCollatorForCLM:
    """Dynamic Padding for CLM.
    
    For CLM, the targets are just the inputs shifted by 1.
    """
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: list[torch.Tensor]) -> dict:
        max_len = max(len(ex) for ex in examples)
        batch_size = len(examples)
        
        input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        for i, ex in enumerate(examples):
            seq_len = len(ex)
            input_ids[i, :seq_len] = ex
            labels[i, :seq_len] = ex
            
        return {'input_ids': input_ids, 'labels': labels}


def train_gpt2(
    data_paths: list[str] = ["data/processed/train.mr"],  # Target language only
    tokenizer_path: str = "data/tokenizer/bpe_32k.model",
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints/gpt2_pretrain",
    resume: bool = False
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1] Preparing Data...")
    tokenizer = Tokenizer(tokenizer_path)
    dataset = CLMDataset(data_paths, tokenizer, max_len=128)
    print(f"Total CLM examples: {len(dataset):,}")
    
    collator = DataCollatorForCLM(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=collator, num_workers=2, pin_memory=True)
    
    print("\n[2] Initializing GPT-2 Model...")
    model = CustomGPT2(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    start_epoch = 0
    history = {'loss': []}
    
    latest_ckpt = checkpoint_dir / 'latest_model.pt'
    if resume:
        ckpt = safe_load_checkpoint(latest_ckpt, device)
        if ckpt is not None:
            print(f"Resuming from {latest_ckpt}")
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            history = ckpt.get('history', {'loss': []})
        else:
            print("No valid checkpoint found. Starting from scratch.")
    
    print(f"\n[3] Starting Pretraining for {epochs} epochs...")
    model.train()
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda', dtype=torch.float16):
                # GPT-2 forward pass
                logits = model(input_ids)
                
                # Shift so that tokens < n predict n
                # logits: [batch, seq_len-1, vocab]
                shift_logits = logits[..., :-1, :].contiguous()
                # labels: [batch, seq_len-1]
                shift_labels = labels[..., 1:].contiguous()
                
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = total_loss / len(dataloader)
        history['loss'].append(avg_loss)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")
        
        safe_save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'history': history,
        }, latest_ckpt)
        print(f"  Checkpoint saved safely.")
        
    print("\n[OK] GPT-2 Pretraining Finished!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    train_gpt2(batch_size=args.batch_size, epochs=args.epochs, resume=args.resume)
