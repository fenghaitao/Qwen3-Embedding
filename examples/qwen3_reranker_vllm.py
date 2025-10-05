import logging

import json
import logging

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from typing import Union, List, Tuple, Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data._utils.worker import ManagerWatchdog

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, is_torch_npu_available
logger = logging.getLogger(__name__)
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import math
from sentence_transformers import CrossEncoder, SentenceTransformer
from vllm.inputs.data import TokensPrompt


class Qwen3Rerankervllm(CrossEncoder):
    def __init__(self, model_name_or_path, instruction="Given the user query, retrieval the relevant passages", **kwargs):
        number_of_gpu=torch.cuda.device_count()
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.max_length=kwargs.get('max_length', 8192)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.sampling_params = SamplingParams(temperature=0, 
            top_p=0.95, 
            max_tokens=1,
            logprobs=20, 
            allowed_token_ids=[self.true_token,self.false_token],
        )
        self.lm = LLM(model=model_name_or_path, tensor_parallel_size=number_of_gpu, max_model_len=10000, enable_prefix_caching=True, distributed_executor_backend='ray', gpu_memory_utilization=0.8)

        

    def format_instruction(self, instruction, query, doc):
        if isinstance(query, tuple):
            instruction = query[0]
            query = query[1]
        text = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        return text

    def compute_scores(self, pairs, **kwargs):
        messages = [self.format_instruction(self.instruction, query, doc) for query, doc in pairs]
        messages =  self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ele[:self.max_length] + self.suffix_tokens for ele in messages]
        messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
        outputs = self.lm.generate(messages, self.sampling_params, use_tqdm=False)
        scores = []
        for i in range(len(outputs)):
            final_logits = outputs[i].outputs[0].logprobs[-1]
            token_count = len(outputs[i].outputs[0].token_ids)
            if self.true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[self.true_token].logprob
            if self.false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)

        return scores

    def stop(self):
        destroy_model_parallel()

if __name__ == '__main__':
    model = Qwen3Rerankervllm(model_name_or_path='Qwen/Qwen3-Reranker-0.6B', instruction="Retrieval document that can answer user's query", max_length=2048)
    try:
        queries = ['What is the capital of China?', 'Explain gravity']
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
        ]
        pairs = list(zip(queries, documents))
        new_scores = model.compute_scores(pairs)
        print('scores', new_scores)
    finally:
        model.stop()
        # Explicitly delete the model to ensure cleanup
        del model



