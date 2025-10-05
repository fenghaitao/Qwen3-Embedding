#coding:utf8
from typing import Dict, Optional, List, Union
import torch

print("Checking CUDA availability...")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    try:
        import vllm
        from vllm import LLM, PoolingParams
        from vllm.distributed.parallel_state import destroy_model_parallel
        use_vllm = True
        print("vLLM is available, using vLLM backend")
    except ImportError:
        use_vllm = False
        print("vLLM not available, using Transformers backend")
else:
    use_vllm = False
    print("CUDA not available, using Transformers backend")

if not use_vllm:
    # Fallback to transformers
    import torch.nn.functional as F
    from torch import Tensor
    from transformers import AutoTokenizer, AutoModel

class Qwen3EmbeddingFallback():
    def __init__(self, model_name_or_path, instruction=None, max_length=8192):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        self.instruction = instruction
        self.use_vllm = use_vllm and cuda_available
        
        if self.use_vllm:
            print("Initializing vLLM backend...")
            self.model = LLM(model=model_name_or_path, task="embed", hf_overrides={"is_matryoshka": True})
        else:
            print("Initializing Transformers backend...")
            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True, 
                torch_dtype=torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True, 
                padding_side='left'
            )
        self.max_length = max_length

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
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
        
        if self.use_vllm:
            if dim > 0:
                output = self.model.embed(sentences, pooling_params=PoolingParams(dimensions=dim))
            else:
                output = self.model.embed(sentences)
            output = torch.tensor([o.outputs.embedding for o in output])
            return output
        else:
            # Transformers fallback
            inputs = self.tokenizer(sentences, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            with torch.no_grad():
                model_outputs = self.model(**inputs)
                output = self.last_token_pool(model_outputs.last_hidden_state, inputs['attention_mask'])
                if dim != -1:
                    output = output[:, :dim]
                output = F.normalize(output, p=2, dim=1)
            return output

    def stop(self):
        if self.use_vllm:
            destroy_model_parallel()

if __name__ == "__main__":
    model_path = "Qwen/Qwen3-Embedding-0.6B"
    model = Qwen3EmbeddingFallback(model_path)
    
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
    
    model.stop()
    print("Example completed successfully!")