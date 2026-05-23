"""
SentencePiece BPE Tokenizer Training & Loading
===============================================
We train a JOINT tokenizer on both Hindi and Marathi text.

Why SentencePiece?
- Language-agnostic: works directly on raw text, no pre-tokenization needed
- Handles Devanagari Unicode natively
- Implements BPE (Byte Pair Encoding) algorithm
- Used by most modern NLP models (T5, LLaMA, etc.)

Why joint vocab?
- Hindi and Marathi share Devanagari script
- Shared subwords enable cross-lingual transfer
- Single vocab simplifies the encoder-decoder architecture in Part II
"""

import sentencepiece as spm
import os
from pathlib import Path


def train_tokenizer(
    input_file: str = "data/processed/all_text_for_tokenizer.txt",
    model_prefix: str = "data/tokenizer/bpe_32k",
    vocab_size: int = 32000,
    character_coverage: float = 0.9999,
    model_type: str = "bpe",
):
    """Train a SentencePiece BPE tokenizer.
    
    Args:
        input_file: Path to text file with one sentence per line
        model_prefix: Output prefix for .model and .vocab files
        vocab_size: Target vocabulary size (32K is standard)
        character_coverage: Fraction of characters to cover (0.9999 for Devanagari)
        model_type: 'bpe' or 'unigram'
    """
    out_dir = Path(model_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training {model_type.upper()} tokenizer...")
    print(f"  Input: {input_file}")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Character coverage: {character_coverage}")
    
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        # Special tokens
        pad_id=0,        # <pad> = 0
        unk_id=1,        # <unk> = 1
        bos_id=2,        # <s>   = 2 (beginning of sentence)
        eos_id=3,        # </s>  = 3 (end of sentence)
        # Additional special tokens for BERT MLM
        user_defined_symbols=["[MASK]"],
        # Training parameters
        num_threads=os.cpu_count(),
        shuffle_input_sentence=True,
        # Normalization
        normalization_rule_name="nfkc",
        # Byte fallback for unknown characters
        byte_fallback=True,
    )
    
    print(f"  Saved: {model_prefix}.model")
    print(f"  Saved: {model_prefix}.vocab")
    
    # Verify
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    print(f"\n  Actual vocab size: {sp.get_piece_size():,}")
    
    # Test tokenization
    test_hi = "दिल्ली अपने आप में एक पूरा पर्यटन स्थल है"
    test_mr = "दिल्ली खरोखरीच एक परिपूर्ण पर्यटन स्थळ आहे"
    
    hi_tokens = sp.encode(test_hi, out_type=str)
    mr_tokens = sp.encode(test_mr, out_type=str)
    hi_ids = sp.encode(test_hi, out_type=int)
    
    print(f"\n  Test (Hindi):   {test_hi}")
    print(f"  Tokens:         {hi_tokens}")
    print(f"  IDs:            {hi_ids}")
    print(f"  Decoded:        {sp.decode(hi_ids)}")
    print(f"\n  Test (Marathi): {test_mr}")
    print(f"  Tokens:         {mr_tokens}")
    
    print("\n[OK] Tokenizer training complete!")
    return sp


class Tokenizer:
    """Wrapper around SentencePiece for easy use in training pipelines."""
    
    def __init__(self, model_path: str = "data/tokenizer/bpe_32k.model"):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.pad_id = self.sp.pad_id()     # 0
        self.unk_id = self.sp.unk_id()     # 1
        self.bos_id = self.sp.bos_id()     # 2
        self.eos_id = self.sp.eos_id()     # 3
        self.mask_id = self.sp.piece_to_id("[MASK]")
        self.vocab_size = self.sp.get_piece_size()
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Encode text to token IDs with optional BOS/EOS."""
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        # Filter out special tokens for clean output
        clean_ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(clean_ids)
    
    def encode_batch(self, texts: list[str], add_bos: bool = True, add_eos: bool = True) -> list[list[int]]:
        """Encode a batch of texts."""
        return [self.encode(t, add_bos, add_eos) for t in texts]
    
    def __len__(self):
        return self.vocab_size


if __name__ == "__main__":
    train_tokenizer()
