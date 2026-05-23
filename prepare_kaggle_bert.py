import os
import zipfile
from pathlib import Path

def create_bert_lstm_archive():
    """Create a lightweight zip for the LSTM+BERT embeddings experiment.
    Only needs src code, data, and tokenizer (no checkpoints).
    """
    print("Preparing BERT LSTM Kaggle Archive...")
    
    output_filename = "adivaani_bert_lstm.zip"
    
    targets = [
        "src/",
        "data/processed/",
        "data/tokenizer/",
    ]
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for target in targets:
            target_path = Path(target)
            if target_path.is_dir():
                for root, dirs, files in os.walk(target):
                    if '__pycache__' in root:
                        continue
                    for file in files:
                        if file.endswith('.pyc'):
                            continue
                        file_path = os.path.join(root, file)
                        print(f"Adding: {file_path}")
                        zipf.write(file_path)
            else:
                print(f"Warning: {target} not found!")
                
    size_mb = os.path.getsize(output_filename) / (1024*1024)
    print(f"\nDone! Created {output_filename} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    create_bert_lstm_archive()
