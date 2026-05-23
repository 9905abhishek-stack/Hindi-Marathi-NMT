"""
Final Test Evaluation Script for Fusion Model
=============================================
Loads the best checkpoint from Kaggle and evaluates on the test set.
Generates training plots and sample translations.
"""

import json
import torch
from pathlib import Path
from src.models.fusion import TranslationFusionModel
from src.data.dataset import get_dataloaders
from src.data.tokenizer import Tokenizer
from src.evaluation.metrics import translate_and_evaluate
from src.evaluation.plots import plot_training_curves

def evaluate_fusion(
    experiment_name: str = "fusion_hi2mr",
    checkpoint_dir: str = "checkpoints",
    embed_dim: int = 768,
    num_layers: int = 6,
    num_heads: int = 12,
    num_kv_heads: int = 4,
    max_len: int = 100,
    direction: str = "hi2mr",
    device: str = "cuda",
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
    model = TranslationFusionModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_ratio=4,
        dropout=0.0,
        pad_id=tokenizer.pad_id
    ).to(device)
    
    ckpt_path = ckpt_dir / 'best_model.pt'
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / 'latest_model.pt'
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded from epoch {ckpt.get('epoch', 0)+1}")
    
    # Evaluate with greedy
    print("\nGreedy decoding evaluation...")
    greedy_bleu, greedy_chrf, hyps, refs = translate_and_evaluate(
        model, test_loader, tokenizer, device, use_beam=False, max_samples=1000
    )
    print(f"  Greedy BLEU:   {greedy_bleu:.2f}")
    print(f"  Greedy CHRF++: {greedy_chrf:.2f}")
    
    # Save results
    results = {
        'experiment': experiment_name,
        'direction': direction,
        'greedy_bleu': greedy_bleu,
        'greedy_chrf': greedy_chrf,
        'best_epoch': ckpt.get('epoch', 0) + 1,
    }
    with open(ckpt_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save example translations
    if hyps and refs:
        examples = []
        for i in range(min(50, len(hyps))):
            examples.append({'hypothesis': hyps[i], 'reference': refs[i]})
        with open(ckpt_dir / 'test_examples.json', 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        print(f"\n--- Sample Translations ---")
        for i in range(min(5, len(hyps))):
            print(f"\n  Ref: {refs[i]}")
            print(f"  Hyp: {hyps[i]}")
    
    # Generate plots
    print("\nGenerating training plots...")
    history = ckpt.get('history', {})
    if history:
        with open(ckpt_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        plot_training_curves(
            str(ckpt_dir / 'history.json'),
            output_dir=str(ckpt_dir / 'plots'),
            experiment_name=experiment_name,
        )
    
    print(f"\n{'='*60}")
    print(f"  DONE! Results saved to {ckpt_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    evaluate_fusion()
