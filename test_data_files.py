#!/usr/bin/env python3
"""Quick test to verify data files are accessible"""

from pathlib import Path
from src.config.settings import RAW_DATA_DIR
from src.data_ingestion.kaggle_downloader import KaggleDownloader

print("=" * 80)
print("DATA FILES VERIFICATION TEST")
print("=" * 80)

# Test 1: Check RAW_DATA_DIR
print(f"\n1. RAW_DATA_DIR: {RAW_DATA_DIR}")
print(f"   Exists: {RAW_DATA_DIR.exists()}")
print(f"   Is directory: {RAW_DATA_DIR.is_dir()}")

# Test 2: List all .txt files
txt_files = list(RAW_DATA_DIR.glob("*.txt"))
print(f"\n2. Found {len(txt_files)} .txt files in RAW_DATA_DIR:")
for f in sorted(txt_files)[:10]:
    print(f"   - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

# Test 3: Check required files
required_files = ["train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
                  "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt"]
print(f"\n3. Checking required files:")
for fname in required_files:
    fpath = RAW_DATA_DIR / fname
    status = "✓ EXISTS" if fpath.exists() else "✗ MISSING"
    print(f"   {status}: {fname}")

# Test 4: Test KaggleDownloader
print(f"\n4. Testing KaggleDownloader:")
downloader = KaggleDownloader(RAW_DATA_DIR)
files = downloader.get_downloaded_files()
print(f"   Downloader found {len(files)} files")

file_info = downloader.get_file_info()
print(f"   File info total: {file_info['total_files']}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

