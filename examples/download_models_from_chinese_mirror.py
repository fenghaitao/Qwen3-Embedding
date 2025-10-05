#!/usr/bin/env python
#coding:utf8
"""
Script to download Qwen3-Embedding-4B and Qwen3-Reranker-4B models from Chinese HF mirror
"""

import os
import sys
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path
import shutil


def download_embedding_model():
    """Download Qwen3-Embedding-4B model files from Chinese mirror"""
    print("Downloading Qwen3-Embedding-4B from Chinese mirror...")
    
    # Set the mirror environment variable
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # Also set token to empty to avoid auth issues
    os.environ['HF_TOKEN'] = ''
    
    model_name = "Qwen/Qwen3-Embedding-4B"
    local_dir = "./models/Qwen3-Embedding-4B"
    
    try:
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Initialize API with mirror endpoint
        api = HfApi(endpoint='https://hf-mirror.com')
        
        # List files in the repository
        print(f"Listing files in {model_name}...")
        files = api.list_repo_files(repo_id=model_name, repo_type="model")
        
        # Filter for important model files
        model_files = [f for f in files if any(ext in f for ext in ['.json', '.bin', '.safetensors', '.txt', '.py', '.md']) or 'config' in f.lower()]
        
        print(f"Found {len(model_files)} files to download")
        
        # Download each file individually
        for file_name in model_files:
            print(f"Downloading {file_name}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=model_name,
                    filename=file_name,
                    cache_dir=local_dir,
                    endpoint='https://hf-mirror.com',
                    resume_download=True
                )
                print(f"Downloaded {file_name} to {downloaded_path}")
            except Exception as e:
                print(f"Warning: Failed to download {file_name}: {e}")
                continue
                
        print("Qwen3-Embedding-4B downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading Qwen3-Embedding-4B: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_reranker_model():
    """Download Qwen3-Reranker-4B model files from Chinese mirror"""
    print("Downloading Qwen3-Reranker-4B from Chinese mirror...")
    
    # Set the mirror environment variable
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # Also set token to empty to avoid auth issues
    os.environ['HF_TOKEN'] = ''
    
    model_name = "Qwen/Qwen3-Reranker-4B"
    local_dir = "./models/Qwen3-Reranker-4B"
    
    try:
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Initialize API with mirror endpoint
        api = HfApi(endpoint='https://hf-mirror.com')
        
        # List files in the repository
        print(f"Listing files in {model_name}...")
        files = api.list_repo_files(repo_id=model_name, repo_type="model")
        
        # Filter for important model files
        model_files = [f for f in files if any(ext in f for ext in ['.json', '.bin', '.safetensors', '.txt', '.py', '.md']) or 'config' in f.lower()]
        
        print(f"Found {len(model_files)} files to download")
        
        # Download each file individually
        for file_name in model_files:
            print(f"Downloading {file_name}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=model_name,
                    filename=file_name,
                    cache_dir=local_dir,
                    endpoint='https://hf-mirror.com',
                    resume_download=True
                )
                print(f"Downloaded {file_name} to {downloaded_path}")
            except Exception as e:
                print(f"Warning: Failed to download {file_name}: {e}")
                continue
                
        print("Qwen3-Reranker-4B downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading Qwen3-Reranker-4B: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Starting download of Qwen3-Embedding-4B and Qwen3-Reranker-4B from Chinese HF mirror...")
    
    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    
    # Download both models
    embedding_success = download_embedding_model()
    print("\n" + "="*50 + "\n")
    reranker_success = download_reranker_model()
    
    print("\nDownload process completed!")
    
    if embedding_success and reranker_success:
        print("All models downloaded successfully!")
        return 0
    else:
        print("Some downloads failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())