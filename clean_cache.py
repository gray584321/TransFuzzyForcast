#!/usr/bin/env python
"""
clean_cache.py

Deletes the feature extraction cache so that all data will be reprocessed.
"""

import os
import shutil
import config  # Ensure your config module defines config.output_dir

def clean_cache():
    cache_dir = os.path.join(config.output_dir, 'cache')
    if os.path.exists(cache_dir):
        confirm = input(f"Are you sure you want to delete the cache directory '{cache_dir}'? (y/n): ")
        if confirm.lower() == 'y':
            shutil.rmtree(cache_dir)
            print("Cache directory has been deleted.")
        else:
            print("Operation cancelled. Cache directory was not deleted.")
    else:
        print("Cache directory does not exist.")

if __name__ == "__main__":
    clean_cache() 