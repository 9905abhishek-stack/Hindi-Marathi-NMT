"""
Final Test Evaluation Script
=============================
Loads the best checkpoint and evaluates on the test set with beam search.
Also generates training plots.
"""

import json
import torch
from pathlib import Path
from src.models.lstm_seq2seq import Seq2Seq
from src.data.dataset import get_dataloaders
from src.data.tokenizer import Tokenizer
from src.evaluation.metrics import translate_and_evaluate
from src.evaluation.plots import plot_training_curves


def evaluate_final(
    experiment_name: str = "lstm_random_hi2mr",
    checkpoint_dir: str = "checkpoints",
    embed_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 2,
    max_len: int = 80,
    direction: str = "hi2mr",
    device: str = "cuda",
    use_beam: bool = True,
    beam_width: int = 5,
):
    ckpt_dir = Path(checkpoint_dir) / experiment_name
    
    print(f"\n{'='*60}")
    print(f"  Final Evaluation: {experiment_name}")
    print(f"{'='*60}")
    
    # Load data
    print("\nLoading data...")
    tokenizer = Tokenizer("data/tokenizer/bpe_32k.model")
    _, _, test_loader, _ = get_dataloaders(
        data_dir="data/processed", tokenizer_path="data/tokenizer/bpe_32k.model",
        batch_size=32, max_len=max_len, direction=direction,
    )
    
    # Load model
    print("Loading best model...")
    model = Seq2Seq(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, padding_idx=tokenizer.pad_id,
    ).to(device)
    
    ckpt = torch.load(ckpt_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded from epoch {ckpt['epoch']+1}, val BLEU: {ckpt['val_bleu']:.2f}")
    
    # Evaluate with greedy first (fast)
    print("\nGreedy decoding evaluation...")
    greedy_bleu, greedy_chrf, _, _ = translate_and_evaluate(
        model, test_loader, tokenizer, device, use_beam=False,
    )
    print(f"  Greedy BLEU-100:   {greedy_bleu:.2f}")
    print(f"  Greedy CHRF++-100: {greedy_chrf:.2f}")
    
    # Beam search (slower but better - limited to 500 samples for speed)
    if use_beam:
        print(f"\nBeam search (width={beam_width}) evaluation on 500 samples...")
        beam_bleu, beam_chrf, hyps, refs = translate_and_evaluate(
            model, test_loader, tokenizer, device,
            use_beam=True, beam_width=beam_width, max_samples=500,
        )
        print(f"  Beam BLEU-100:   {beam_bleu:.2f}")
        print(f"  Beam CHRF++-100: {beam_chrf:.2f}")
    else:
        beam_bleu, beam_chrf = greedy_bleu, greedy_chrf
        hyps, refs = None, None
    
    # Save results
    results = {
        'experiment': experiment_name,
        'direction': direction,
        'greedy_bleu': greedy_bleu,
        'greedy_chrf': greedy_chrf,
        'beam_bleu': beam_bleu,
        'beam_chrf': beam_chrf,
        'beam_width': beam_width,
        'best_epoch': ckpt['epoch'] + 1,
        'val_bleu': ckpt['val_bleu'],
    }
    with open(ckpt_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save example translations
    if hyps and refs:
        examples = []
        for i in range(min(30, len(hyps))):
            examples.append({'hypothesis': hyps[i], 'reference': refs[i]})
        with open(ckpt_dir / 'test_examples.json', 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        print(f"\n--- Sample Translations ---")
        for i in range(min(5, len(hyps))):
            print(f"\n  Ref: {refs[i]}")
            print(f"  Hyp: {hyps[i]}")
    
    # Generate plots
    print("\nGenerating training plots...")
    plot_training_curves(
        str(ckpt_dir / 'history.json'),
        output_dir=str(ckpt_dir / 'plots'),
        experiment_name=experiment_name,
    )
    
    print(f"\n{'='*60}")
    print(f"  DONE! Results saved to {ckpt_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    evaluate_final()
