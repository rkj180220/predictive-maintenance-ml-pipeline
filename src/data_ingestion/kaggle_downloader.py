"""
Kaggle Dataset Downloader for NASA C-MAPSS
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
import subprocess
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class KaggleDownloader:
    """
    Download NASA C-MAPSS dataset from Kaggle
    Handles authentication and dataset download
    """

    def __init__(self, download_dir: Path):
        """
        Initialize Kaggle downloader

        Args:
            download_dir: Directory to download datasets
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.kaggle_dataset = "behrad3d/nasa-cmaps"

    def check_kaggle_credentials(self) -> bool:
        """
        Check if Kaggle credentials are configured
        Checks both ~/.kaggle/kaggle.json and .env file

        Returns:
            bool: True if credentials exist, False otherwise
        """
        kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"

        # First check for kaggle.json file
        if kaggle_json_path.exists():
            logger.info("Kaggle credentials found in ~/.kaggle/kaggle.json")
            return True

        # Fallback to .env file
        kaggle_username = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")

        if kaggle_username and kaggle_key:
            logger.info("Kaggle credentials found in .env file")
            # Set environment variables for kaggle CLI
            os.environ["KAGGLE_USERNAME"] = kaggle_username
            os.environ["KAGGLE_KEY"] = kaggle_key
            return True

        # No credentials found
        logger.warning("Kaggle credentials not found")
        logger.info("Please set up Kaggle credentials using one of these methods:")
        logger.info("Method 1 - kaggle.json file:")
        logger.info("  1. Go to https://www.kaggle.com/account")
        logger.info("  2. Click 'Create New API Token'")
        logger.info("  3. Save kaggle.json to ~/.kaggle/kaggle.json")
        logger.info("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        logger.info("Method 2 - .env file:")
        logger.info("  Add to .env file:")
        logger.info("    KAGGLE_USERNAME=your_username")
        logger.info("    KAGGLE_KEY=your_api_key")
        return False

    def download_dataset(self, force_download: bool = False) -> bool:
        """
        Download NASA C-MAPSS dataset from Kaggle

        Args:
            force_download: Force re-download even if files exist

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check credentials
            if not self.check_kaggle_credentials():
                logger.error("Cannot download without Kaggle credentials")
                return False

            # Check if already downloaded
            if not force_download and self._check_existing_files():
                logger.info("Dataset files already exist. Use force_download=True to re-download")
                return True

            logger.info(f"Downloading dataset: {self.kaggle_dataset}")
            logger.info(f"Download directory: {self.download_dir}")

            # Download using kaggle CLI
            cmd = [
                "kaggle", "datasets", "download",
                "-d", self.kaggle_dataset,
                "-p", str(self.download_dir),
                "--unzip"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info("Dataset downloaded successfully")
            logger.debug(f"Download output: {result.stdout}")

            # Handle subdirectory structure (files may be in CMaps subdirectory)
            self._organize_files()

            # Verify downloaded files
            if self._verify_downloaded_files():
                logger.info("All required files verified")
                return True
            else:
                logger.error("Some required files are missing")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Kaggle download failed: {e.stderr}")
            logger.error(f"Command output: {e.stdout}")
            return False
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False

    def _organize_files(self):
        """
        Organize files by moving them from subdirectories to the main download directory
        The Kaggle dataset may extract files into a CMaps subdirectory
        """
        import shutil

        # Check for CMaps subdirectory
        cmaps_dir = self.download_dir / "CMaps"

        if cmaps_dir.exists() and cmaps_dir.is_dir():
            logger.info("Found CMaps subdirectory, moving files to main directory...")

            # Move all .txt files from CMaps to parent directory
            for file_path in cmaps_dir.glob("*.txt"):
                target_path = self.download_dir / file_path.name
                if target_path.exists():
                    logger.debug(f"File {file_path.name} already exists, skipping")
                else:
                    shutil.move(str(file_path), str(target_path))
                    logger.debug(f"Moved {file_path.name} to {self.download_dir}")

            # Remove empty CMaps directory if all files moved
            try:
                if not any(cmaps_dir.glob("*.txt")):
                    logger.info("Cleaning up CMaps subdirectory...")
                    # Keep the directory but remove .txt files have been moved
            except Exception as e:
                logger.warning(f"Could not clean up CMaps directory: {e}")
        else:
            logger.debug("No CMaps subdirectory found, files are in correct location")

    def _check_existing_files(self) -> bool:
        """Check if dataset files already exist"""
        required_patterns = [
            "train_FD002.txt",
            "test_FD002.txt",
            "RUL_FD002.txt",
            "train_FD004.txt",
            "test_FD004.txt",
            "RUL_FD004.txt"
        ]

        existing_files = list(self.download_dir.glob("*.txt"))
        existing_names = [f.name for f in existing_files]

        for pattern in required_patterns:
            if pattern not in existing_names:
                return False

        return True

    def _verify_downloaded_files(self) -> bool:
        """Verify that all required files were downloaded"""
        return self._check_existing_files()

    def get_downloaded_files(self) -> List[Path]:
        """
        Get list of downloaded dataset files

        Returns:
            List of file paths
        """
        return sorted(self.download_dir.glob("*.txt"))

    def get_file_info(self) -> dict:
        """
        Get information about downloaded files

        Returns:
            Dictionary with file information
        """
        files = self.get_downloaded_files()
        info = {
            "total_files": len(files),
            "files": []
        }

        for file_path in files:
            file_stat = file_path.stat()
            info["files"].append({
                "name": file_path.name,
                "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                "path": str(file_path)
            })

        return info


def setup_kaggle_credentials(username: str, api_key: str):
    """
    Setup Kaggle credentials programmatically

    Args:
        username: Kaggle username
        api_key: Kaggle API key
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    kaggle_json = kaggle_dir / "kaggle.json"

    credentials = {
        "username": username,
        "key": api_key
    }

    with open(kaggle_json, 'w', encoding='utf-8') as f:
        json.dump(credentials, f, ensure_ascii=False)

    # Set proper permissions
    os.chmod(kaggle_json, 0o600)

    logger.info("Kaggle credentials saved successfully")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data" / "raw"

    downloader = KaggleDownloader(data_dir)

    if downloader.download_dataset():
        print("\nDownloaded files:")
        file_info = downloader.get_file_info()
        for file in file_info["files"]:
            print(f"  - {file['name']} ({file['size_mb']} MB)")
    else:
        print("\nDataset download failed. Please check Kaggle credentials.")
