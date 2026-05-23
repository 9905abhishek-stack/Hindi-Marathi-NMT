"""
Safe Checkpointing Utilities
=============================
Provides atomic, crash-safe checkpoint saving and loading.

Problem: If power is cut while torch.save() is writing a file, the file
gets corrupted — it's half-written bytes. On resume, torch.load() will crash.

Solution: Write to a TEMPORARY file first, then RENAME (atomic on all OSes).
Renaming is an atomic filesystem operation — it either happens completely
or not at all. So the checkpoint file is always either the old valid version
or the new valid version, never a half-written mess.
"""

import os
import json
import shutil
import hashlib
import torch
from pathlib import Path


def safe_save_checkpoint(state: dict, filepath: str | Path):
    """Atomically save a PyTorch checkpoint.
    
    Steps:
    1. Save to a temporary file (filepath.tmp)
    2. Verify the temp file is readable
    3. Rename temp -> final (atomic operation)
    
    If power is cut during step 1, the old checkpoint is still intact.
    If power is cut during step 3, the rename either completed or didn't.
    """
    filepath = Path(filepath)
    tmp_path = filepath.with_suffix('.tmp')
    
    # Step 1: Save to temporary file
    torch.save(state, tmp_path)
    
    # Step 2: Verify the temp file can be loaded (catches disk errors)
    try:
        _ = torch.load(tmp_path, map_location='cpu', weights_only=False)
    except Exception as e:
        # Temp file is corrupt — delete it, keep the old checkpoint
        tmp_path.unlink(missing_ok=True)
        print(f"  [WARNING] Checkpoint verification failed: {e}")
        print(f"  [WARNING] Keeping previous checkpoint intact.")
        return False
    
    # Step 3: Atomic rename (replaces old file safely)
    if os.name == 'nt':  # Windows
        # Windows os.rename fails if target exists, so remove first
        # This creates a tiny window of vulnerability, but it's the best we can do
        filepath.unlink(missing_ok=True)
    
    tmp_path.rename(filepath)
    return True


def safe_save_json(data: dict, filepath: str | Path):
    """Atomically save a JSON file."""
    filepath = Path(filepath)
    tmp_path = filepath.with_suffix('.json.tmp')
    
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Verify
    try:
        with open(tmp_path, 'r', encoding='utf-8') as f:
            json.load(f)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        return False
    
    if os.name == 'nt':
        filepath.unlink(missing_ok=True)
    tmp_path.rename(filepath)
    return True


def safe_load_checkpoint(filepath: str | Path, device: str = 'cpu'):
    """Safely load a checkpoint, with fallback detection.
    
    Returns:
        checkpoint dict, or None if corrupt/missing
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return None
    
    try:
        ckpt = torch.load(filepath, map_location=device, weights_only=False)
        # Basic integrity checks
        assert 'model_state_dict' in ckpt, "Missing model_state_dict"
        assert 'epoch' in ckpt, "Missing epoch"
        return ckpt
    except Exception as e:
        print(f"  [WARNING] Checkpoint {filepath} is corrupt: {e}")
        
        # Check if a backup exists
        backup = filepath.with_suffix('.bak')
        if backup.exists():
            try:
                ckpt = torch.load(backup, map_location=device, weights_only=False)
                print(f"  [RECOVERED] Loaded backup from {backup}")
                return ckpt
            except Exception:
                print(f"  [FAILED] Backup is also corrupt. Starting fresh.")
        
        return None


def get_checkpoint_info(checkpoint_dir: str | Path) -> dict:
    """Get a summary of checkpoint state for status reporting."""
    checkpoint_dir = Path(checkpoint_dir)
    
    info = {
        'has_latest': (checkpoint_dir / 'latest_model.pt').exists(),
        'has_best': (checkpoint_dir / 'best_model.pt').exists(),
        'has_history': (checkpoint_dir / 'history.json').exists(),
    }
    
    if info['has_history']:
        try:
            with open(checkpoint_dir / 'history.json', 'r') as f:
                h = json.load(f)
            info['completed_epochs'] = len(h.get('train_loss', []))
            info['latest_train_loss'] = h['train_loss'][-1] if h['train_loss'] else None
            info['latest_val_loss'] = h['val_loss'][-1] if h.get('val_loss') else None
            info['best_val_bleu'] = max(h['val_bleu']) if h.get('val_bleu') and any(v > 0 for v in h['val_bleu']) else None
        except Exception:
            info['completed_epochs'] = 0
    
    if info['has_latest']:
        try:
            ckpt = torch.load(checkpoint_dir / 'latest_model.pt', map_location='cpu', weights_only=False)
            info['latest_epoch'] = ckpt.get('epoch', -1)
        except Exception:
            info['latest_epoch'] = -1
    
    return info
