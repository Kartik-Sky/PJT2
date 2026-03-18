from transformers import PretrainedConfig

from config.ModelSettings import CMSConfig
from config.supported import SUPPORTED_MODELS

class NoraConfig(PretrainedConfig):

    model_type = "NORA"

    def __init__(
            self,
        model_name: str = "llama-7b",
        custom_model_path: str = None,
        dtype: str = "fp16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,
        max_seq_length: int = 4096,
        # Pass your CMSConfig fields here too so everything is serializable
        cms_cfg : CMSConfig = None,
        num_hidden_layers=12,   # 🔴 REQUIRED
        num_attention_heads=12, # 🔴 REQUIRED (used in cache shapes)
        device = "auto",
        **kwargs,
    ):
        self.model_name = model_name
        self.custom_model_path = custom_model_path
        self.model_dtype = dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flash_attention = use_flash_attention
        self.max_seq_length = max_seq_length
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # if cms_cfg is None: 
        #     raise ValueError("cms_config not found")
        self.device = device
        self.cms_cfg = cms_cfg
        super().__init__(**kwargs)
