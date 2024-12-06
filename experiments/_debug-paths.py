# debug_paths.py
import os
import pandas as pd
from pathlib import Path

def check_paths(val_csv, base_dir):
    # Read validation CSV
    df = pd.read_csv(val_csv)
    
    print(f"Base directory: {base_dir}")
    print(f"Total images in CSV: {len(df)}\n")
    
    # Check each path
    for idx, row in df.iterrows():
        full_path = os.path.join(base_dir, row['filename'])
        path_exists = os.path.exists(full_path)
        
        print(f"Image {idx + 1}:")
        print(f"Filename: {row['filename']}")
        print(f"Full path: {full_path}")
        print(f"Exists: {path_exists}")
        if not path_exists:
            # Check parent directories
            parent = Path(full_path).parent
            print(f"Parent dir exists: {parent.exists()}")
            if parent.exists():
                print(f"Parent dir contents: {list(parent.iterdir())}")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    config_base_dir = "/home/zehra.korkusuz/ClipBased-SyntheticImageDetection/data"
    config_val_csv = os.path.join(config_base_dir, "debug_val.csv")
    
    check_paths(config_val_csv, config_base_dir)