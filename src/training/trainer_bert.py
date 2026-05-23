"""
BERT Pretraining Pipeline (Masked Language Modeling)
====================================================
Trains the custom BERT encoder on the combined Hindi-Marathi corpus
using Masked Language Modeling (MLM).
"""

import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

from src.models.transformer import CustomBERT
from src.data.tokenizer import Tokenizer
from src.training.checkpoint_utils import safe_save_checkpoint, safe_load_checkpoint


class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling.
    
    Reads raw text lines, tokenizes them, and chunks them into max_len.
    We combine Hindi and Marathi text to pretrain a cross-lingual BERT.
    """
    def __init__(self, file_paths: list[str], tokenizer: Tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        
        print(f"Loading MLM data from {file_paths}...")
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = self.tokenizer.encode(line)
                    # Chunk long sequences
                    for i in range(0, len(tokens), max_len):
                        chunk = tokens[i:i + max_len]
                        if len(chunk) >= 5:  # Skip very short chunks
                            self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class DataCollatorForMLM:
    """Dynamic Masking for MLM.
    
    Masks 15% of the tokens for prediction.
    Of the masked tokens:
    - 80% are replaced with [MASK]
    - 10% are replaced with a random token
    - 10% are kept unchanged
    """
    def __init__(self, tokenizer: Tokenizer, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples: list[torch.Tensor]) -> dict:
        # Pad sequences to max length in batch
        max_len = max(len(ex) for ex in examples)
        batch_size = len(examples)
        
        input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        for i, ex in enumerate(examples):
            seq_len = len(ex)
            input_ids[i, :seq_len] = ex
            
            # Create masking matrix (ignore special tokens)
            prob_matrix = torch.full((seq_len,), self.mlm_probability)
            
            # Don't mask special tokens
            special_tokens_mask = [
                1 if t in [self.tokenizer.pad_id, self.tokenizer.bos_id, self.tokenizer.eos_id] else 0 
                for t in ex.tolist()
            ]
            prob_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            
            masked_indices = torch.bernoulli(prob_matrix).bool()
            
            # Labels only for masked tokens, -100 otherwise
            labels[i, :seq_len][masked_indices] = ex[masked_indices]
            
            # 80% of the time: replace with [MASK]
            indices_replaced = torch.bernoulli(torch.full((seq_len,), 0.8)).bool() & masked_indices
            input_ids[i, :seq_len][indices_replaced] = self.tokenizer.mask_id
            
            # 10% of the time: replace with random word
            indices_random = torch.bernoulli(torch.full((seq_len,), 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), (seq_len,), dtype=torch.long)
            input_ids[i, :seq_len][indices_random] = random_words[indices_random]
            
            # The remaining 10% are kept unchanged
            
        return {'input_ids': input_ids, 'labels': labels}


def train_bert(
    data_paths: list[str] = ["data/processed/train.hi", "data/processed/train.mr"],
    tokenizer_path: str = "data/tokenizer/bpe_32k.model",
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints/bert_pretrain",
    resume: bool = False
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1] Preparing Data...")
    tokenizer = Tokenizer(tokenizer_path)
    dataset = MLMDataset(data_paths, tokenizer, max_len=128)
    print(f"Total MLM examples: {len(dataset):,}")
    
    collator = DataCollatorForMLM(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=collator, num_workers=2, pin_memory=True)
    
    print("\n[2] Initializing BERT Model...")
    model = CustomBERT(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id).to(device)
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
                logits = model(input_ids)
                # logits: [batch, seq_len, vocab], labels: [batch, seq_len]
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
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
        
    print("\n[OK] BERT Pretraining Finished!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    train_bert(batch_size=args.batch_size, epochs=args.epochs, resume=args.resume)
