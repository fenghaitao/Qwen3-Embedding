#coding:utf8
import os
from typing import Dict, Optional, List, Union
import torch
from torch import nn
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
from collections import defaultdict

class Qwen3Embedding():
    def __init__(self, model_name_or_path, instruction=None, use_fp16: bool = False, use_cuda: bool = False, max_length=8192):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        self.instruction = instruction
        
        print(f"Loading model from: {model_name_or_path}")
        try:
            # Use CPU mode and fp32 for compatibility
            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True, 
                torch_dtype=torch.float32,
                device_map=None  # Load on CPU
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True, 
                padding_side='left'
            )
            print("Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        self.max_length = max_length
    
    def last_token_pool(self, last_hidden_states: Tensor,
        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        if task_description is None:
            task_description = self.instruction
        return f'Instruct: {task_description}\nQuery:{query}'

    def encode(self, sentences: Union[List[str], str], is_query: bool = False, instruction=None, dim: int = -1):
        if isinstance(sentences, str):
            sentences = [sentences]
        if is_query:
            sentences = [self.get_detailed_instruct(instruction, sent) for sent in sentences]
        
        print(f"Encoding {len(sentences)} sentences...")
        inputs = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        
        with torch.no_grad():
            model_outputs = self.model(**inputs)
            output = self.last_token_pool(model_outputs.last_hidden_state, inputs['attention_mask'])
            if dim != -1:
                output = output[:, :dim]
            output = F.normalize(output, p=2, dim=1)
        return output

if __name__ == "__main__":
    # Try with a smaller model first or fallback to a different model
    model_path = "Qwen/Qwen3-Embedding-0.6B"
    
    print("Initializing Qwen3 Embedding model...")
    try:
        model = Qwen3Embedding(model_path, use_cuda=False, use_fp16=False)
        
        queries = ['What is the capital of China?', 'Explain gravity']
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
        ]
        
        dim = 1024
        print("Encoding queries...")
        query_outputs = model.encode(queries, is_query=True, dim=dim)
        print("Encoding documents...")
        doc_outputs = model.encode(documents, dim=dim)
        
        print('Query outputs shape:', query_outputs.shape)
        print('Document outputs shape:', doc_outputs.shape)
        
        scores = (query_outputs @ doc_outputs.T) * 100
        print("Similarity scores:", scores.tolist())
        print("Example completed successfully!")
        
    except Exception as e:
        print(f"Error running example: {e}")
        print("\nThis might be due to:")
        print("1. Network issues downloading the model")
        print("2. Model compatibility issues")
        print("3. Missing dependencies")
        print("\nYou may need to:")
        print("1. Check your internet connection")
        print("2. Try using HF_HUB_CACHE environment variable") 
        print("3. Manually download the model files")