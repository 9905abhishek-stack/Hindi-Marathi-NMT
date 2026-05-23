"""
Evaluation Metrics: BLEU-100 and CHRF++-100
============================================
Uses sacrebleu for standardized, reproducible metric computation.
All scores reported on 0-100 scale as required by the assignment.
"""

import sacrebleu
from src.data.tokenizer import Tokenizer


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score on 0-100 scale.
    
    BLEU measures n-gram precision (n=1..4) with brevity penalty.
    We use sacrebleu for reproducibility (no custom tokenization needed).
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score  # Already on 0-100 scale


def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus-level CHRF++ score on 0-100 scale.
    
    CHRF++ uses character n-grams (1-6) + word n-grams (1-2).
    Better than BLEU for morphologically rich languages like Hindi/Marathi.
    """
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return chrf.score  # Already on 0-100 scale


def translate_and_evaluate(model, dataloader, tokenizer: Tokenizer, 
                           device: str = 'cuda', max_samples: int = None,
                           use_beam: bool = False, beam_width: int = 5):
    """Translate all sentences in dataloader and compute metrics.
    
    Args:
        model: Seq2Seq model
        dataloader: DataLoader with source/target pairs
        tokenizer: Tokenizer for decoding IDs to text
        device: 'cuda' or 'cpu'
        max_samples: Limit number of samples (for validation during training)
        use_beam: Use beam search instead of greedy
        beam_width: Beam width for beam search
    
    Returns:
        bleu: BLEU-100 score
        chrf: CHRF++-100 score  
        hypotheses: List of translated strings
        references: List of reference strings
    """
    import torch
    model.eval()
    
    hypotheses = []
    references = []
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src_ids = batch['src_ids'].to(device)
            src_lengths = batch['src_lengths'].to(device)
            tgt_ids = batch['tgt_ids']
            
            batch_size = src_ids.size(0)
            
            if use_beam:
                # Beam search: one at a time
                for i in range(batch_size):
                    gen = model.translate_beam(
                        src_ids[i:i+1], src_lengths[i:i+1],
                        beam_width=beam_width,
                        bos_id=tokenizer.bos_id,
                        eos_id=tokenizer.eos_id,
                    )
                    hyp = tokenizer.decode(gen[0].cpu().tolist())
                    ref = tokenizer.decode(tgt_ids[i].tolist())
                    hypotheses.append(hyp)
                    references.append(ref)
            else:
                # Greedy decoding: whole batch at once
                generated = model.translate_greedy(
                    src_ids, src_lengths,
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                )
                
                for i in range(batch_size):
                    hyp = tokenizer.decode(generated[i].cpu().tolist())
                    ref = tokenizer.decode(tgt_ids[i].tolist())
                    hypotheses.append(hyp)
                    references.append(ref)
            
            total += batch_size
            if max_samples and total >= max_samples:
                break
    
    # Compute metrics
    bleu = compute_bleu(hypotheses, references)
    chrf = compute_chrf(hypotheses, references)
    
    return bleu, chrf, hypotheses, references
