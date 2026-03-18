from config.CustomCLMConfig import NoraConfig
from config.ModelSettings import CMSConfig
from transformers import AutoModelForCausalLM
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
# from config.ModelSettings import NoraConfig
from model.CausalLM import NoraCausalLM

cms_config = CMSConfig(3072, 3, [1,2,3], [3,4,2], nn.GELU)

config = NoraConfig(model_name="llama-3.2-3b", cms_cfg=cms_config)
print("\n\n",config.dtype, "\n\n")

AutoConfig.register("NORA", NoraConfig)
AutoModelForCausalLM.register(NoraConfig, NoraCausalLM)

model = AutoModelForCausalLM.from_config(config)

# ---- Save & reload ----
# model.save_pretrained("./nora-checkpoint")

# model = AutoModelForCausalLM.from_pretrained("./nora-checkpoint")

# ---- Generation (beam search, sampling, etc. — all work) ----
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
inputs = tokenizer("Hello, I am Nora", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.decode(output[0]))
