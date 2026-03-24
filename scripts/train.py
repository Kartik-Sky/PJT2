import torch
import numpy as np 
import datasets
from model.CausalLM import NoraCausalLM
from config.CustomCLMConfig import NoraConfig
from config.ModelSettings import CMSConfig
from transformers import AutoModelForCausalLM
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
# from config.ModelSettings import NoraConfig
from model.CausalLM import NoraCausalLM
from tqdm import tqdm
from optimizers.AdamW import CMSAdamW
from torch.utils.data import DataLoader

import os
from torch.utils.tensorboard import SummaryWriter

# Directories
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

log_dir = "./runs/nora_training_2"
writer = SummaryWriter(log_dir=log_dir)


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

cms_config = CMSConfig(3072, 3, [1,2,3], [3,4,2], nn.GELU)

config = NoraConfig(model_name="llama-3.2-3b", cms_cfg=cms_config)

num_epochs = 20

AutoConfig.register("NORA", NoraConfig)
AutoModelForCausalLM.register(NoraConfig, NoraCausalLM)

model = AutoModelForCausalLM.from_config(config)
checkpoint = torch.load(
    "/root/KartikGoyal/checkpoints/checkpoint_epoch_7.pt",
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
print("Dataset loaded")


def param_gen(param_dict):
    for block_name, (frequency, block_params) in param_dict.items():
        flat = []
        for p in block_params:
            if isinstance(p, (list, tuple)):
                flat.extend(p)
            else:
                flat.append(p)

        yield {
            "params": flat,
            "block": block_name,
            "frequency": frequency
        }

def tokenize(example):

    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=2048
    )

    return {"input_ids": tokens["input_ids"], "attention_mask":tokens["attention_mask"]}

dataset = dataset.map(tokenize, batched=True)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
print("Dataset tokenized and formatted")


train_loader = DataLoader(
    dataset["train"],
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print("DataLoader created")

params = model.nora.cms.get_param_groups()
# optimizer = CMSAdamW(param_gen(params), 1e-4)
optimizer = CMSAdamW(param_gen(params), lr = 1e-4)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
global_step = 0

model.train()
for epoch in range(num_epochs):

    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device) 
        attention_mask = batch['attention_mask'].to(device) 

        labels = input_ids.clone() 
        
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = output.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Logging
        writer.add_scalar("Loss/train_step", loss.item(), global_step)

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        avg_loss = total_loss / num_batches

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}',
        })

    # Epoch-level logging
    writer.add_scalar("Loss/train_epoch", avg_loss, epoch)

    print(f"\nEpoch {epoch+1}/{num_epochs} complete — Avg Loss: {avg_loss:.4f}\n")

    # -----------------------
    # CHECKPOINTING
    # -----------------------
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")

    torch.save({
        'epoch': epoch + 8,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config,
    }, checkpoint_path)

    print(f"Checkpoint saved at: {checkpoint_path}")