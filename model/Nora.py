import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple, Dict
from config.supported import SUPPORTED_MODELS, SUPPORTED_DTYPES
from config.ModelSettings import CMSConfig
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoTokenizer,
    PreTrainedConfig,
)
from memory_module.cms import CMSNet
from Exceptions import ModelNotSupportedError

class Nora(nn.Module):

    def __init__(self, 
                 model_name = "llama-7b", 
                 custom_model_path = None, 
                 device: str = "auto",
                dtype: str = "fp16",
                load_in_8bit: bool = False,
                load_in_4bit: bool = False,
                use_flash_attention: bool = False,
                max_seq_length: int = 4096
                 ):
        
        super().__init__()
                # ---- Validation ----
        if custom_model_path is None and model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Choose from: {list(SUPPORTED_MODELS.keys())} "
                f"or provide a 'custom_model_path'."
            )

        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Choose from: {list(SUPPORTED_DTYPES.keys())}"
            )

        if load_in_8bit and load_in_4bit:
            raise ValueError("Cannot enable both 8-bit and 4-bit quantization simultaneously.")

        if device not in ("auto", "cpu", "cuda") and not device.startswith("cuda:"):
            raise ValueError(
                f"Unsupported device '{device}'. Use 'auto', 'cpu', 'cuda', or 'cuda:<id>'."
            )

        if device != "auto" and device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")

        self.model_name: str = model_name
        self.model_path: str = custom_model_path or SUPPORTED_MODELS[model_name]
        self.device_str: str = device
        self.dtype: torch.dtype = SUPPORTED_DTYPES[dtype]
        self.load_in_8bit: bool = load_in_8bit
        self.load_in_4bit: bool = load_in_4bit
        self.use_flash_attention: bool = use_flash_attention
        self.max_seq_length: int = max_seq_length
        
        self.model: Optional[LlamaForCausalLM] = None
        self.tokenizer = None
        self.config: Optional[LlamaConfig] = None

    def _load_tokenizer(self, tokenizerConfig: Optional[PreTrainedConfig] = None)->None:
        """Load the tokenizer corresponding to the chosen model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, config = tokenizerConfig)
        except:
            raise Exception("Error Loading Tokenizer")
        

    def _load_model(self, config: CMSConfig):
        """Load the model corresponding to the chosen model and add the CMS"""
        
        if (self.model_name is not None):
            foundation_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        elif (self.model_path is not None):
            try:
                foundation_model = torch.load(self.model_path)
            except:
                raise ModelNotSupportedError(f"Failed to load the model at {self.model_path}.")
            
        else:
                raise ModelNotSupportedError("Failed to initialize foundation_model. Please check the model_name or model_path")
        
        self.cms = CMSNet(config = config)
        self.decoder = foundation_model.get_decoder()
        
        # Freezing the Parameters of the Decoder Network
        for param in self.decoder.parameters():
            param.requires_grad = False
        
        self.lm_head = foundation_model.lm_head
    
    def forward(self, input_ids, attention_mask, use_cache):

        x = self.decoder(input_ids = input_ids, attention_mask = attention_mask, use_cache= use_cache)
        x = self.cms(x)
        return self.lm_head(x)
