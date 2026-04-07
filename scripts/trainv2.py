import torch
import os
import datasets
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Custom Imports (Assuming these remain in your path)
from model.CausalLM import NoraCausalLM
from config.CustomCLMConfig import NoraConfig
from config.ModelSettings import CMSConfig
from optimizers.AdamW import CMSAdamW

# 1. SETUP DIRECTORIES
# Using a unified experiment folder for better organization
EXP_NAME = "nora_experiment_v4"
BASE_DIR = f"./{EXP_NAME}"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs3")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)

# 2. CONFIG & MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cms_config = CMSConfig(4096, 5, [1,1,1,1,1], [4,4,4,4,4], nn.GELU)
cms_config = CMSConfig(
    4096,
    5,
    [1, 2, 4, 8, 16],   # Hierarchical frequencies — slow→fast nesting
    [4, 4, 3, 2, 2],    # Higher expansion early, compressed at fast layers
    nn.SiLU             # SiLU (Swish) > GELU for deep LLMs empirically
)
config = NoraConfig(model_name="llama-3-8b", cms_cfg=cms_config)

AutoConfig.register("NORA", NoraConfig)
AutoModelForCausalLM.register(NoraConfig, NoraCausalLM)

model = AutoModelForCausalLM.from_config(config)

# Load Checkpoint (Updating epoch count based on your logic)
START_EPOCH = 17
checkpoint_path = f"{BASE_DIR}/checkpoints/checkpoint_epoch_{START_EPOCH}.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cuda')
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {START_EPOCH}")
else:
    checkpoint = None
    START_EPOCH = 0
    print("No checkpoint found, starting from scratch.")

model.to(device)

# 3. TOKENIZER & DATASET
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_and_mask(example):
    outputs = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=2048
    )
    
    # FIX: Create labels and mask padding with -100
    # CrossEntropyLoss ignores -100 by default.
    input_ids = torch.tensor(outputs["input_ids"])
    attention_mask = torch.tensor(outputs["attention_mask"])
    
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100 
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

raw_dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
tokenized_dataset = raw_dataset.map(
    tokenize_and_mask, 
    batched=True, 
    remove_columns=raw_dataset["train"].column_names
)
tokenized_dataset.set_format(type="torch")

train_loader = DataLoader(tokenized_dataset["train"], batch_size=10, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(tokenized_dataset["validation"], batch_size=8, shuffle=False, num_workers=4)

# 4. OPTIMIZER
def param_gen(param_dict):
    for block_name, (frequency, block_params) in param_dict.items():
        flat = []
        for p in block_params:
            if isinstance(p, (list, tuple)): flat.extend(p)
            else: flat.append(p)
        yield {"params": flat, "block": block_name, "frequency": frequency}

params = model.nora.cms.get_param_groups()
optimizer = CMSAdamW(param_gen(params), lr=1e-4)
if checkpoint is not None and 'optimizer_state_dict' in checkpoint :
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

torch.cuda.empty_cache()

print(torch.cuda.memory_summary(device=None, abbreviated=False))

# 5. TRAINING LOOP
num_epochs = 20
global_step = 0

for epoch in range(START_EPOCH, START_EPOCH + num_epochs):
    model.train()
    total_train_loss = 0
    train_batches = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for batch in train_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        if torch.isnan(loss):
            continue

        loss.backward()
        optimizer.step()

        # Log step-level
        writer.add_scalar("Loss/train_step", loss.item(), global_step)
        total_train_loss += loss.item()
        global_step += 1
        train_batches += 1
        
        train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = total_train_loss / len(train_loader)

    # 6. VALIDATION LOOP
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        for batch in val_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    
    # 7. LOGGING & CHECKPOINTING
    writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
    writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
    
    print(f"\nEnd of Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

    checkpoint_save_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pt")
    torch.save({
        'epoch': epoch + START_EPOCH + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, checkpoint_save_path)
