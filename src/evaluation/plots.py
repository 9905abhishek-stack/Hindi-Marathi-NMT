"""
Plotting Utilities for Training Metrics
========================================
Generates publication-quality plots for:
- Train/Val loss curves
- Train/Val BLEU-100 curves
- Train/Val CHRF++-100 curves
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curves(history_path: str, output_dir: str = "plots", 
                         experiment_name: str = ""):
    """Generate all required plots from training history."""
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    prefix = f"{experiment_name}_" if experiment_name else ""
    
    # Style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'train': '#2196F3', 'val': '#F44336'}
    
    # ── Loss Plot ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history['train_loss'], color=colors['train'], 
            linewidth=2, label='Train Loss', marker='o', markersize=3)
    ax.plot(epochs, history['val_loss'], color=colors['val'],
            linewidth=2, label='Val Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training & Validation Loss — {experiment_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f'{prefix}loss.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # ── BLEU-100 Plot ──
    if 'train_bleu' in history and 'val_bleu' in history:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, history['train_bleu'], color=colors['train'],
                linewidth=2, label='Train BLEU-100', marker='o', markersize=3)
        ax.plot(epochs, history['val_bleu'], color=colors['val'],
                linewidth=2, label='Val BLEU-100', marker='s', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('BLEU-100', fontsize=12)
        ax.set_title(f'BLEU-100 Score — {experiment_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f'{prefix}bleu.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # ── CHRF++-100 Plot ──
    if 'train_chrf' in history and 'val_chrf' in history:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, history['train_chrf'], color=colors['train'],
                linewidth=2, label='Train CHRF++-100', marker='o', markersize=3)
        ax.plot(epochs, history['val_chrf'], color=colors['val'],
                linewidth=2, label='Val CHRF++-100', marker='s', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('CHRF++-100', fontsize=12)
        ax.set_title(f'CHRF++-100 Score — {experiment_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f'{prefix}chrf.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Plots saved to {output_dir}/")


def plot_comparison(history_paths: dict, output_dir: str = "plots"):
    """Compare multiple experiments side-by-side.
    
    Args:
        history_paths: {'Random Embeddings': 'path/to/history.json', 'BERT Embeddings': '...'}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
    
    for metric_key, metric_name, fname in [
        ('val_bleu', 'Validation BLEU-100', 'comparison_bleu.png'),
        ('val_chrf', 'Validation CHRF++-100', 'comparison_chrf.png'),
        ('val_loss', 'Validation Loss', 'comparison_loss.png'),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for i, (name, path) in enumerate(history_paths.items()):
            with open(path, 'r') as f:
                h = json.load(f)
            epochs = list(range(1, len(h[metric_key]) + 1))
            ax.plot(epochs, h[metric_key], color=colors[i % len(colors)],
                    linewidth=2, label=name, marker='o', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Comparison plots saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        plot_training_curves(sys.argv[1], experiment_name=sys.argv[2] if len(sys.argv) > 2 else "")
