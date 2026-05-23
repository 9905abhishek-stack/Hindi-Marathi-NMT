"""
Quick Training Status Checker
==============================
Run this anytime to see the progress of all training experiments.

Usage:
    python -m src.training.status
"""

from pathlib import Path
from src.training.checkpoint_utils import get_checkpoint_info


def check_all():
    checkpoint_root = Path("checkpoints")
    
    if not checkpoint_root.exists():
        print("No checkpoints directory found.")
        return
    
    experiments = sorted([d for d in checkpoint_root.iterdir() if d.is_dir()])
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"\n{'='*65}")
    print(f"  TRAINING STATUS REPORT")
    print(f"{'='*65}")
    
    for exp_dir in experiments:
        info = get_checkpoint_info(exp_dir)
        name = exp_dir.name
        
        print(f"\n  [{name}]")
        
        if info['has_latest']:
            epochs_done = info.get('completed_epochs', '?')
            latest_epoch = info.get('latest_epoch', -1) + 1
            print(f"    Completed epochs:  {epochs_done}")
            print(f"    Checkpoint epoch:  {latest_epoch}")
        else:
            print(f"    Status: NOT STARTED")
            continue
        
        if info.get('latest_train_loss') is not None:
            print(f"    Last train loss:   {info['latest_train_loss']:.4f}")
        if info.get('latest_val_loss') is not None:
            print(f"    Last val loss:     {info['latest_val_loss']:.4f}")
        if info.get('best_val_bleu') is not None:
            print(f"    Best val BLEU:     {info['best_val_bleu']:.2f}")
        
        print(f"    Has best model:    {'Yes' if info['has_best'] else 'No'}")
    
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    check_all()
