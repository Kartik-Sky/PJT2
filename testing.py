from config.CustomCLMConfig import NoraConfig
from config.ModelSettings import CMSConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from model.CausalLM import NoraCausalLM
import torch
import torch.nn as nn

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Config ----
cms_config = CMSConfig(3072, 3, [1,2,3], [3,4,2], nn.GELU)
model_config = NoraConfig(model_name="llama-3.2-3b", cms_cfg=cms_config)

# ---- Register ----
AutoConfig.register("NORA", NoraConfig)
AutoModelForCausalLM.register(NoraConfig, NoraCausalLM)

# ---- Init model ----
model = AutoModelForCausalLM.from_config(model_config)

# ---- Load checkpoint ----
checkpoint = torch.load(
    "./nora_experiment_v2/checkpoints/checkpoint_epoch_2.pt",
    map_location=device
)
print(checkpoint.keys())
model.load_state_dict(checkpoint["model_state_dict"])

# ---- Move model to GPU ----
model.to(device)
model.eval()

print("Model Loaded")

# ---- Tokenizer ----
print("Loading Tokenizer")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
print("Tokenizer Loaded")

# ---- Generation ----
print("Generating Output")

inputs = tokenizer("Hello, I am Nora", return_tensors="pt")

# Move inputs to same device
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))