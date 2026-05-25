import os
import zipfile
from pathlib import Path

def create_kaggle_archive():
    print("Preparing Kaggle Upload Archive...")
    
    output_filename = "hindi_marathi_nmt_kaggle.zip"
    
    # Directories and files to include
    targets = [
        "src/",
        "data/processed/",
        "data/tokenizer/",
        "checkpoints/bert_pretrain/",
        "checkpoints/gpt2_pretrain/",
    ]
    
    # Optional files
    if os.path.exists("requirements.txt"):
        targets.append("requirements.txt")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for target in targets:
            target_path = Path(target)
            if target_path.is_file():
                print(f"Adding file: {target}")
                zipf.write(target)
            elif target_path.is_dir():
                for root, dirs, files in os.walk(target):
                    # Skip __pycache__
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
                
    print(f"\nDone! Successfully created {output_filename}")
    print(f"Size: {os.path.getsize(output_filename) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    create_kaggle_archive()
