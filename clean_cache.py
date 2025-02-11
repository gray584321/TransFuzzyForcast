#!/usr/bin/env python
"""
clean_cache.py

Removes cached pickle files to force recomputation of features and model training.
"""

import os
import glob
import argparse

def clean_cache(directory):
    """
    Removes specified cache files from the given directory.
    """
    print(f"Cleaning cache in directory: {directory}")

    # Define the patterns for cache files to be removed.  Crucially,
    # we now include *both* cache files.
    cache_patterns = [
        "processed_features.pkl",
        "feature_extraction_cache.pkl",
        "feature_scaler.pkl",
        "best_model.pth",
        "trained_model_weights.pth"
    ]

    files_removed = 0
    for pattern in cache_patterns:
        # Use glob to find all files matching the pattern
        files_to_remove = glob.glob(os.path.join(directory, pattern))

        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
                files_removed += 1
            except OSError as e:
                print(f"Error removing {file_path}: {e}")

    if files_removed == 0:
        print("No cache files found to remove.")
    else:
        print(f"Total cache files removed: {files_removed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean cache files.")
    parser.add_argument('--dir', type=str, default='results/',
                        help='Directory to clean cache files from. Defaults to "results/".')
    args = parser.parse_args()

    clean_cache(args.dir)
    print("Cache cleaning completed.") 