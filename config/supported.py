from typing import Dict
import torch

SUPPORTED_MODELS: Dict[str, str] = {
    "llama-7b": "meta-llama/Llama-2-7b-hf",
    "llama-13b": "meta-llama/Llama-2-13b-hf",
    "llama-70b": "meta-llama/Llama-2-70b-hf",
    "llama-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
}


SUPPORTED_DTYPES: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}