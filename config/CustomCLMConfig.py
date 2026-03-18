from transformers import PretrainedConfig

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
        cms_hidden_size: int = 512,
        cms_num_layers: int = 4,
        **kwargs,
    ):
        self.model_name = model_name
        self.custom_model_path = custom_model_path
        self.dtype = dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flash_attention = use_flash_attention
        self.max_seq_length = max_seq_length
        self.cms_hidden_size = cms_hidden_size
        self.cms_num_layers = cms_num_layers
        super().__init__(**kwargs)
