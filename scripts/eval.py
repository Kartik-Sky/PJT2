from tqdm import tqdm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM
from model.CausalLM import NoraCausalLM
from config.CustomCLMConfig import NoraConfig
from config.ModelSettings import CMSConfig
import torch.nn as nn

dataset = load_dataset("wikitext", "wikitext-2-v1")
test_texts = dataset["test"]["text"]



def preprocess(texts, tokenizer, max_length=512):
    texts = [t for t in texts if len(t.strip()) > 0]

    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    return encodings

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

encodings = preprocess(test_texts, tokenizer)

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

dataset = LMDataset(encodings)
loader = DataLoader(dataset, batch_size=8)

def evaluate(model, dataloader, device="cuda"):
    model.eval()
    model = model.to(device)

    total_loss = 0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # ---- Shift ----
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]
            attn = attention_mask[:, :-1]

            outputs = model(
                input_ids=inputs,
                attention_mask=attn
            )

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]

            # ---- Flatten ----
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)

            loss = F.cross_entropy(
                logits,
                labels,
                ignore_index=tokenizer.pad_token_id,
                reduction="sum"
            )

            batch_tokens = (labels != tokenizer.pad_token_id).sum().item()

            total_loss += loss.item()
            total_tokens += batch_tokens

            # ---- Update tqdm ----
            progress_bar.set_postfix({
                "avg_loss": f"{(total_loss / max(total_tokens,1)):.4f}",
                "tokens": total_tokens
            })

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

cms_config = CMSConfig(3072, 3, [1,2,3], [3,4,2], nn.GELU)

config = NoraConfig(model_name="llama-3.2-3b", cms_cfg=cms_config)

AutoConfig.register("NORA", NoraConfig)
AutoModelForCausalLM.register(NoraConfig, NoraCausalLM)
model = AutoModelForCausalLM.from_config(config)
for i in range(1,17):

    checkpoint = torch.load(
        f"/root/KartikGoyal/checkpoints/checkpoint_epoch_{i}.pt",
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)


    loss, ppl = evaluate(model, loader, device=device)
    
    print("Checkpoint Number: ",i)
    print(f"Test Loss: {loss:.4f}")
    print(f"Perplexity: {ppl:.3f}")