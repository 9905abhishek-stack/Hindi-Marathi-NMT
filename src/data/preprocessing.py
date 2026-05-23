"""
Data Preprocessing Pipeline for Hindi-Marathi Parallel Corpus
=============================================================
Steps:
1. Load raw parallel text files (train.hi, train.mr, test.hi, test.mr)
2. Unicode NFC normalization (standardizes Devanagari combining characters)
3. Clean whitespace (extra spaces in the corpus between words)
4. Filter bad pairs (empty, too long, extreme length ratios)
5. Split training data into train/val sets
6. Save processed data
"""

import unicodedata
import re
import random
import json
import os
from pathlib import Path
from collections import Counter


def normalize_unicode(text: str) -> str:
    """Apply NFC normalization to standardize Devanagari characters.
    
    NFC (Canonical Decomposition followed by Canonical Composition) ensures that
    characters like 'क' + '्' + 'ष' are consistently represented. Without this,
    the same visual character could have multiple byte representations, confusing
    the tokenizer and model.
    """
    return unicodedata.normalize('NFC', text)


def clean_whitespace(text: str) -> str:
    """Remove excessive whitespace commonly found in the corpus.
    
    The raw data has inconsistent spacing like 'इस  प्रयोग  को  नित्य' 
    (double/triple spaces between words). We normalize to single spaces.
    """
    text = re.sub(r'\s+', ' ', text)  # Collapse all whitespace to single space
    return text.strip()


def is_valid_pair(src: str, tgt: str, max_len: int = 200, max_ratio: float = 3.0) -> bool:
    """Filter out bad sentence pairs.
    
    Args:
        src: Source sentence
        tgt: Target sentence
        max_len: Maximum word count for either sentence
        max_ratio: Maximum allowed length ratio between source and target
    
    Returns:
        True if the pair passes all quality checks
    """
    if not src or not tgt:
        return False
    
    src_words = src.split()
    tgt_words = tgt.split()
    
    # Skip empty after split
    if len(src_words) == 0 or len(tgt_words) == 0:
        return False
    
    # Skip extremely long sentences (they slow down training and are often noisy)
    if len(src_words) > max_len or len(tgt_words) > max_len:
        return False
    
    # Skip pairs with extreme length ratios (likely misaligned)
    ratio = max(len(src_words), len(tgt_words)) / max(min(len(src_words), len(tgt_words)), 1)
    if ratio > max_ratio:
        return False
    
    # Skip very short pairs (< 2 words) - not useful for learning
    if len(src_words) < 2 or len(tgt_words) < 2:
        return False
    
    return True


def load_parallel_file(filepath: str) -> list[str]:
    """Load a text file with one sentence per line."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def compute_statistics(hi_sents: list[str], mr_sents: list[str], label: str = ""):
    """Compute and print dataset statistics."""
    hi_lens = [len(s.split()) for s in hi_sents]
    mr_lens = [len(s.split()) for s in mr_sents]
    
    stats = {
        'num_pairs': len(hi_sents),
        'hi_avg_len': sum(hi_lens) / max(len(hi_lens), 1),
        'hi_max_len': max(hi_lens) if hi_lens else 0,
        'mr_avg_len': sum(mr_lens) / max(len(mr_lens), 1),
        'mr_max_len': max(mr_lens) if mr_lens else 0,
    }
    
    print(f"\n{'='*50}")
    print(f"Dataset Statistics: {label}")
    print(f"{'='*50}")
    print(f"  Number of pairs : {stats['num_pairs']:,}")
    print(f"  Hindi  avg len  : {stats['hi_avg_len']:.1f} words")
    print(f"  Hindi  max len  : {stats['hi_max_len']} words")
    print(f"  Marathi avg len : {stats['mr_avg_len']:.1f} words")
    print(f"  Marathi max len : {stats['mr_max_len']} words")
    print(f"{'='*50}\n")
    
    return stats


def preprocess_and_split(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    val_ratio: float = 0.05,
    seed: int = 42,
    max_len: int = 200,
    max_ratio: float = 3.0,
):
    """Full preprocessing pipeline.
    
    1. Load raw files
    2. Clean and normalize
    3. Filter bad pairs
    4. Split train into train/val
    5. Save processed files
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load raw data ──
    print("Loading raw data...")
    train_hi = load_parallel_file(raw_dir / "train.hi")
    train_mr = load_parallel_file(raw_dir / "train.mr")
    test_hi = load_parallel_file(raw_dir / "test.hi")
    test_mr = load_parallel_file(raw_dir / "test.mr")
    
    assert len(train_hi) == len(train_mr), f"Train mismatch: {len(train_hi)} vs {len(train_mr)}"
    assert len(test_hi) == len(test_mr), f"Test mismatch: {len(test_hi)} vs {len(test_mr)}"
    
    print(f"  Raw train pairs: {len(train_hi):,}")
    print(f"  Raw test pairs:  {len(test_hi):,}")
    
    # ── Clean and filter ──
    print("\nCleaning and filtering...")
    clean_hi, clean_mr = [], []
    filtered_count = 0
    
    for hi, mr in zip(train_hi, train_mr):
        hi_clean = clean_whitespace(normalize_unicode(hi))
        mr_clean = clean_whitespace(normalize_unicode(mr))
        
        if is_valid_pair(hi_clean, mr_clean, max_len=max_len, max_ratio=max_ratio):
            clean_hi.append(hi_clean)
            clean_mr.append(mr_clean)
        else:
            filtered_count += 1
    
    print(f"  Kept: {len(clean_hi):,} pairs")
    print(f"  Filtered: {filtered_count:,} pairs")
    
    # ── Split into train/val ──
    print(f"\nSplitting (val_ratio={val_ratio})...")
    random.seed(seed)
    indices = list(range(len(clean_hi)))
    random.shuffle(indices)
    
    val_size = int(len(indices) * val_ratio)
    val_indices = set(indices[:val_size])
    
    final_train_hi, final_train_mr = [], []
    final_val_hi, final_val_mr = [], []
    
    for i in range(len(clean_hi)):
        if i in val_indices:
            final_val_hi.append(clean_hi[i])
            final_val_mr.append(clean_mr[i])
        else:
            final_train_hi.append(clean_hi[i])
            final_train_mr.append(clean_mr[i])
    
    # ── Clean test data ──
    final_test_hi, final_test_mr = [], []
    for hi, mr in zip(test_hi, test_mr):
        hi_clean = clean_whitespace(normalize_unicode(hi))
        mr_clean = clean_whitespace(normalize_unicode(mr))
        if hi_clean and mr_clean:
            final_test_hi.append(hi_clean)
            final_test_mr.append(mr_clean)
    
    # ── Save processed data ──
    def save_lines(filepath, lines):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    
    save_lines(out_dir / "train.hi", final_train_hi)
    save_lines(out_dir / "train.mr", final_train_mr)
    save_lines(out_dir / "val.hi", final_val_hi)
    save_lines(out_dir / "val.mr", final_val_mr)
    save_lines(out_dir / "test.hi", final_test_hi)
    save_lines(out_dir / "test.mr", final_test_mr)
    
    # ── Statistics ──
    train_stats = compute_statistics(final_train_hi, final_train_mr, "Train")
    val_stats = compute_statistics(final_val_hi, final_val_mr, "Validation")
    test_stats = compute_statistics(final_test_hi, final_test_mr, "Test")
    
    # Save stats JSON
    all_stats = {'train': train_stats, 'val': val_stats, 'test': test_stats}
    with open(out_dir / "stats.json", 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)
    
    # ── Save combined text for tokenizer training ──
    print("Saving combined text for tokenizer training...")
    all_text = final_train_hi + final_train_mr
    save_lines(out_dir / "all_text_for_tokenizer.txt", all_text)
    print(f"  Total sentences for tokenizer: {len(all_text):,}")
    
    print("\n[OK] Preprocessing complete!")
    return all_stats


if __name__ == "__main__":
    preprocess_and_split()
