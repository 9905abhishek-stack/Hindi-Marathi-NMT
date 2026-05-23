"""
PyTorch Dataset and DataLoader for Hindi-Marathi Translation
=============================================================
Provides:
- TranslationDataset: Loads parallel text, tokenizes, returns padded tensors
- collate_fn: Dynamic batching with padding to max length in batch
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.data.tokenizer import Tokenizer


class TranslationDataset(Dataset):
    """Dataset for parallel Hindi-Marathi sentence pairs.
    
    Each sample returns:
        src_ids: Token IDs for source sentence (with BOS/EOS)
        tgt_ids: Token IDs for target sentence (with BOS/EOS)
    """
    
    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        tokenizer: Tokenizer,
        max_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load sentences
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sents = [line.strip() for line in f if line.strip()]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sents = [line.strip() for line in f if line.strip()]
        
        assert len(self.src_sents) == len(self.tgt_sents), \
            f"Source/target mismatch: {len(self.src_sents)} vs {len(self.tgt_sents)}"
    
    def __len__(self):
        return len(self.src_sents)
    
    def __getitem__(self, idx):
        src_ids = self.tokenizer.encode(self.src_sents[idx], add_bos=True, add_eos=True)
        tgt_ids = self.tokenizer.encode(self.tgt_sents[idx], add_bos=True, add_eos=True)
        
        # Truncate to max_len
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]
        
        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
        }


def collate_fn(batch, pad_id: int = 0):
    """Collate function with dynamic padding.
    
    Pads all sequences in a batch to the length of the longest sequence
    in that batch (not a global max). This is more efficient because
    shorter batches waste less computation on padding.
    
    Returns:
        src_ids: [batch, max_src_len] - padded source token IDs
        src_lengths: [batch] - actual lengths (for packing in LSTM)
        tgt_ids: [batch, max_tgt_len] - padded target token IDs
        tgt_lengths: [batch] - actual target lengths
    """
    src_ids = [item['src_ids'] for item in batch]
    tgt_ids = [item['tgt_ids'] for item in batch]
    
    src_lengths = torch.tensor([len(s) for s in src_ids], dtype=torch.long)
    tgt_lengths = torch.tensor([len(t) for t in tgt_ids], dtype=torch.long)
    
    # Pad to max length in this batch
    src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)
    
    return {
        'src_ids': src_padded,
        'src_lengths': src_lengths,
        'tgt_ids': tgt_padded,
        'tgt_lengths': tgt_lengths,
    }


def get_dataloaders(
    data_dir: str = "data/processed",
    tokenizer_path: str = "data/tokenizer/bpe_32k.model",
    batch_size: int = 64,
    max_len: int = 128,
    num_workers: int = 0,
    direction: str = "hi2mr",
):
    """Create train/val/test DataLoaders.
    
    Args:
        direction: 'hi2mr' (Hindi->Marathi) or 'mr2hi' (Marathi->Hindi)
    """
    tokenizer = Tokenizer(tokenizer_path)
    data_dir = Path(data_dir)
    
    if direction == "hi2mr":
        src_lang, tgt_lang = "hi", "mr"
    else:
        src_lang, tgt_lang = "mr", "hi"
    
    train_ds = TranslationDataset(
        data_dir / f"train.{src_lang}", data_dir / f"train.{tgt_lang}",
        tokenizer, max_len
    )
    val_ds = TranslationDataset(
        data_dir / f"val.{src_lang}", data_dir / f"val.{tgt_lang}",
        tokenizer, max_len
    )
    test_ds = TranslationDataset(
        data_dir / f"test.{src_lang}", data_dir / f"test.{tgt_lang}",
        tokenizer, max_len
    )
    
    pad_id = tokenizer.pad_id
    collate = lambda batch: collate_fn(batch, pad_id=pad_id)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(batch_size=4)
    batch = next(iter(train_loader))
    print("Source shape:", batch['src_ids'].shape)
    print("Target shape:", batch['tgt_ids'].shape)
    print("Source lengths:", batch['src_lengths'])
    print("Sample decoded src:", tokenizer.decode(batch['src_ids'][0].tolist()))
    print("Sample decoded tgt:", tokenizer.decode(batch['tgt_ids'][0].tolist()))
