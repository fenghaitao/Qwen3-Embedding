import logging
from typing import Dict, Optional, List
import torch

print("Checking CUDA availability for reranker...")
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    try:
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.distributed.parallel_state import destroy_model_parallel
        from vllm.inputs.data import TokensPrompt
        use_vllm = True
        print("vLLM is available, using vLLM backend")
    except ImportError:
        use_vllm = False
        print("vLLM not available, using Transformers backend")
else:
    use_vllm = False
    print("CUDA not available, using Transformers backend")

if not use_vllm:
    # Fallback to transformers implementation
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import math

class Qwen3RerankerFallback:
    def __init__(self, model_name_or_path, instruction="Given the user query, retrieval the relevant passages", **kwargs):
        self.instruction = instruction
        self.use_vllm = use_vllm and cuda_available
        
        if self.use_vllm:
            print("Initializing vLLM reranker backend...")
            number_of_gpu = torch.cuda.device_count()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.max_length = kwargs.get('max_length', 8192)
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
            self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
            self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
            self.sampling_params = SamplingParams(
                temperature=0, 
                top_p=0.95, 
                max_tokens=1,
                logprobs=20, 
                allowed_token_ids=[self.true_token, self.false_token],
            )
            self.lm = LLM(
                model=model_name_or_path, 
                tensor_parallel_size=number_of_gpu, 
                max_model_len=10000, 
                enable_prefix_caching=True, 
                distributed_executor_backend='ray', 
                gpu_memory_utilization=0.8
            )
        else:
            print("Initializing Transformers reranker backend...")
            self.max_length = kwargs.get('max_length', 4096)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True, 
                padding_side='left'
            )
            # Use CPU mode and float32 for compatibility
            self.lm = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True, 
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map=None  # Load on CPU
            ).eval()
            
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            
            self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            
            self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def format_instruction(self, instruction, query, doc):
        if self.use_vllm:
            if isinstance(query, tuple):
                instruction = query[0]
                query = query[1]
            text = [
                {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
                {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
            ]
            return text
        else:
            if instruction is None:
                instruction = self.instruction
            output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
            return output

    def process_inputs(self, pairs):
        """Process inputs for transformers backend"""
        out = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(out['input_ids']):
            out['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        out = self.tokenizer.pad(out, padding=True, return_tensors="pt", max_length=self.max_length)
        return out

    @torch.no_grad()
    def compute_logits_transformers(self, inputs, **kwargs):
        """Compute logits using transformers backend"""
        batch_scores = self.lm(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def compute_logits_vllm(self, model, messages, sampling_params, true_token, false_token):
        """Compute logits using vLLM backend"""
        outputs = model.generate(messages, sampling_params, use_tqdm=False)
        scores = []
        for i in range(len(outputs)):
            final_logits = outputs[i].outputs[0].logprobs[-1]
            if true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[true_token].logprob
            if false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)
        return scores

    def compute_scores(self, pairs, **kwargs):
        if self.use_vllm:
            # vLLM implementation
            messages = [self.format_instruction(self.instruction, query, doc) for query, doc in pairs]
            messages = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
            )
            messages = [ele[:self.max_length] + self.suffix_tokens for ele in messages]
            messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
            scores = self.compute_logits_vllm(self.lm, messages, self.sampling_params, self.true_token, self.false_token)
            return scores
        else:
            # Transformers implementation
            pairs = [self.format_instruction(self.instruction, query, doc) for query, doc in pairs]
            inputs = self.process_inputs(pairs)
            scores = self.compute_logits_transformers(inputs)
            return scores

    def stop(self):
        if self.use_vllm:
            destroy_model_parallel()

if __name__ == '__main__':
    model = Qwen3RerankerFallback(
        model_name_or_path='Qwen/Qwen3-Reranker-0.6B', 
        instruction="Retrieval document that can answer user's query", 
        max_length=2048
    )
    
    queries = ['What is the capital of China?', 'Explain gravity']
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    
    pairs = list(zip(queries, documents))
    print("Computing reranking scores...")
    scores = model.compute_scores(pairs)
    print('Scores:', scores)
    
    model.stop()
    print("Reranker example completed successfully!")