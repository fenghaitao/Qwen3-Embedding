# SWIFT Training Support

The Qwen3-Embedding series models can be further trained when needed (for example, when users have domain-specific data). This document describes how to start training using the [SWIFT framework](https://github.com/modelscope/swift).

ModelScope SWIFT is a large model and multimodal large model training and deployment framework provided by the ModelScope community, with the following characteristics:

- Model Types: Supports 500+ pure text large models and 200+ multimodal large models, covering the entire process from training to deployment.
- Hardware Support: Compatible with CPU, RTX series GPUs, T4/V100, A10/A100/H100, Ascend NPU, MPS, etc.
- Training Methods: Supports full-parameter fine-tuning, LoRA, QLoRA, DoRA, and other techniques.
- Distributed Training: Supports distributed training techniques such as DDP, device_map, DeepSpeed ZeRO-2/ZeRO-3, FSDP, and integrates Megatron's parallel techniques including tensor parallelism, pipeline parallelism, sequence parallelism, and expert parallelism.
- RLHF Training: Supports human alignment methods for both pure text and multimodal large models, such as DPO, GRPO, DAPO, RM, PPO, KTO, etc.

Before starting training, please ensure your environment is properly configured.

```bash
pip install ms-swift -U
# Install from source
pip install git+https://github.com/modelscope/ms-swift.git

pip install transformers -U

# Optional packages
pip install deepspeed # multi-GPU training
pip install liger-kernel # save GPU memory resources
pip install flash-attn --no-build-isolation
```

## Data Preparation

Different loss types affect the dataset input format. Using the following data format as an example:

```json
# sample without rejected_response
{"query": "sentence1", "response": "sentence1-positive"}
# sample with multiple rejected_response
{"query": "sentence1", "response": "sentence1-positive", "rejected_response":  ["sentence1-negative1", "sentence1-negative2", ...]}
```

The above data format is used when the loss is `infonce`. This loss uses `response (positive)` and `query (anchor)` for positive training, and uses `rejected_response (negatives)` for negative training.

## Loss Types

It is recommended to use infonce loss (`--loss_type infonce`) for training.
Infonce loss has several adjustable environment variables:

- INFONCE_TEMPERATURE: Temperature value for the infonce similarity matrix, default is `0.01`.
- INFONCE_USE_BATCH: Whether to use other batches (including samples from other GPUs during DDP) as negatives, default is `True`. If set to `False`, only the rejected_response of the current sample is used as negatives, requiring each sample in the dataset to have at least one negative sample.
- INFONCE_HARD_NEGATIVES: Pads negatives (repeated sampling) or truncates them to ensure the same number of negatives for each sample. Default is `False`.
- INFONCE_MASK_FAKE_NEGATIVE: If negatives exist with similarity greater than positive similarity + `0.1`, mask them to prevent false negatives. Default is `False`.

By default, each row in the dataset can have any number of rejected_responses or none at all.

Other loss types can also be used for training, such as `--loss_type cosine_similarity`, where the dataset format is different:

```json
{"query": "sentence1", "response": "sentence2", "label": 0.8}
```

Under this loss, the label field is a float type marking the similarity value between two sentences.

Other types of losses are also supported. A complete introduction to losses and data formats can be found [here](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Embedding.md).

## Complete Training Command

Using infonce loss as an example, the complete training command is as follows:

```shell
nproc_per_node=8
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen3-Embedding-0.6B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset sentence-transformers/stsb:positive \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --eval_steps 20 \
    --num_train_epochs 5 \
    --save_steps 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --label_names labels \
    --dataloader_drop_last true \
    --deepspeed zero3
```

If you use LoRA for training(`--train_type lora`), you can use SWIFT to merge the LoRA weights back into the model for inference:

```shell
swift export --adapters /path/of/output/ckpt-xxx --merge_lora true
```

## Inference and Deployment

SWIFT's support for inference and deployment of the Qwen3-Embedding series models is currently under development. In the meantime, users can directly use the [inference code from the ModelCard](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#sentence-transformers-usage).

## Some Code Addresses

1. [Embedding dataset encode](https://github.com/modelscope/ms-swift/blob/main/swift/llm/template/base.py#L338)
2. [Embedding dataset data collator](https://github.com/modelscope/ms-swift/blob/main/swift/llm/template/base.py#L1313)
3. [Embedding loss](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py#L272)