#!/usr/bin/env python
#coding:utf8
"""
Script to download Qwen3-Embedding-4B and Qwen3-Reranker-4B models from Chinese HF mirror
"""

import os
from huggingface_hub import snapshot_download


def download_embedding_model():
    """Download Qwen3-Embedding-4B model from Chinese mirror"""
    print("Downloading Qwen3-Embedding-4B from Chinese mirror...")
    
    # Set the mirror environment variable
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    model_name = "Qwen/Qwen3-Embedding-4B"
    
    try:
        # Download the entire model repository
        print(f"Downloading model files for {model_name}...")
        snapshot_download(
            repo_id=model_name,
            local_dir="./models/Qwen3-Embedding-4B"
        )
        print("Qwen3-Embedding-4B downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading Qwen3-Embedding-4B: {e}")


def download_reranker_model():
    """Download Qwen3-Reranker-4B model from Chinese mirror"""
    print("Downloading Qwen3-Reranker-4B from Chinese mirror...")
    
    # Set the mirror environment variable
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    model_name = "Qwen/Qwen3-Reranker-4B"
    
    try:
        # Download the entire model repository
        print(f"Downloading model files for {model_name}...")
        snapshot_download(
            repo_id=model_name,
            local_dir="./models/Qwen3-Reranker-4B"
        )
        print("Qwen3-Reranker-4B downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading Qwen3-Reranker-4B: {e}")


if __name__ == "__main__":
    print("Starting download of Qwen3-Embedding-4B and Qwen3-Reranker-4B from Chinese HF mirror...")
    
    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    
    # Download both models
    download_embedding_model()
    print("\n" + "="*50 + "\n")
    download_reranker_model()
    
    print("\nDownload process completed!")