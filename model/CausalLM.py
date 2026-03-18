from transformers import PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from config.CustomCLMConfig import NoraConfig
from config.ModelSettings import CMSConfig
from Nora import Nora
import torch.nn as nn


class NoraCausalLM(PreTrainedModel):
    config_class = NoraConfig
    
    def __init__(self, config: NoraConfig):
        super().__init__(config)

        foundation = AutoModelForCausalLM.from_pretrained(config.model_path)
        cms_cfg = CMSConfig(hidden_size=config.cms_hidden_size)

        self.nora = Nora(foundation, cms_cfg)
        self.lm_head = self.nora.lm_head  # expose for HF resize_token_embeddings

        self.post_init()

    def get_input_embeddings(self):
        return self.nora.decoder.embed_tokens

    def set_input_embeddings(self, v):
        self.nora.decoder.embed_tokens = v

    def get_output_embeddings(self):
        return self.nora.lm_head

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "past_key_values": past_key_values, "use_cache": kwargs.get("use_cache", True)}

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                labels=None, use_cache=True, return_dict=True, **kwargs):

        # All real logic delegated to your inner class
        hidden, past_key_values = self.nora(input_ids, attention_mask, use_cache)

        logits = self.nora.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1),
            )

        if not return_dict:
            return (loss, logits, past_key_values) if loss else (logits, past_key_values)

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)
