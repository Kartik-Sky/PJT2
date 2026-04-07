"""
smoke_test.py
─────────────
Runs 10 train steps, 5 val steps, one generation log, and one checkpoint
save/reload — all with a tiny in-memory dataset slice so the whole thing
finishes in under 2 minutes. If this exits with "ALL CHECKS PASSED" your
full training script is safe to leave overnight.

Run with:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.smoke_test
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"]     = "expandable_segments:True"

import sys
import tempfile
import multiprocessing

import torch
import torch.nn as nn
import datasets
from tqdm import tqdm
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from model.CausalLM import NoraCausalLM
from config.CustomCLMConfig import NoraConfig
from config.ModelSettings import CMSConfig
from optimizers.AdamW import CMSAdamW

# ── How many steps to run ─────────────────────────────────────────
TRAIN_STEPS = 10
VAL_STEPS   = 5
SEQ_LEN     = 128    # short sequences — we're testing flow, not capacity
BATCH_SIZE  = 2
ACCUM_STEPS = 2


def check(label: str, condition: bool):
    if not condition:
        print(f"  FAIL: {label}")
        sys.exit(1)
    print(f"  OK  : {label}")


def main():
    print("\n=== SMOKE TEST START ===\n")

    # ── Temp dir for all outputs ──────────────────────────────────
    tmp = tempfile.mkdtemp(prefix="nora_smoke_")
    ckpt_dir  = os.path.join(tmp, "checkpoints")
    log_dir   = os.path.join(tmp, "logs")
    gen_dir   = os.path.join(tmp, "gen_logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Temp dir: {tmp}\n")

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    if device.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # ── Model ─────────────────────────────────────────────────────
    print("--- 1. Building model ---")
    cms_config = CMSConfig(4096, 5, [1,2,4,8,16], [4,4,3,2,2], nn.SiLU)
    config     = NoraConfig(model_name="llama-3-8b", cms_cfg=cms_config)

    AutoConfig.register("NORA", NoraConfig)
    AutoModelForCausalLM.register(NoraConfig, NoraCausalLM)

    model = AutoModelForCausalLM.from_config(config)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters: {param_count:.2f}B")
    check("model on correct device", next(model.parameters()).device.type == device.type)

    # ── Tokenizer ─────────────────────────────────────────────────
    print("\n--- 2. Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token
    check("tokenizer loaded", tokenizer is not None)

    # ── Tiny synthetic dataset (no disk I/O needed) ───────────────
    print("\n--- 3. Building tiny dataset ---")

    # Pull 200 rows from wikitext-2 (tiny, fast to download)
    raw = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex["text"] for ex in raw if len(ex["text"].strip()) > 20][:200]

    # Tokenize and pack into fixed-length chunks
    all_ids = []
    for t in texts:
        ids = tokenizer(t, truncation=False, padding=False)["input_ids"]
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id)

    total   = (len(all_ids) // SEQ_LEN) * SEQ_LEN
    all_ids = all_ids[:total]
    chunks  = [all_ids[i:i+SEQ_LEN] for i in range(0, total, SEQ_LEN)]

    dataset = datasets.Dataset.from_dict({
        "input_ids":      chunks,
        "attention_mask": [[1]*SEQ_LEN for _ in chunks],
        "labels":         chunks,
    })
    dataset.set_format(type="torch")

    check("dataset non-empty",          len(dataset) > 0)
    check("input_ids shape correct",    dataset[0]["input_ids"].shape == torch.Size([SEQ_LEN]))
    check("labels shape correct",       dataset[0]["labels"].shape    == torch.Size([SEQ_LEN]))
    check("no padding in dataset",      (dataset[0]["attention_mask"] == 0).sum().item() == 0)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Optimizer ─────────────────────────────────────────────────
    print("\n--- 4. Optimizer ---")

    def param_gen(param_dict):
        for block_name, (frequency, block_params) in param_dict.items():
            flat = []
            for p in block_params:
                if isinstance(p, (list, tuple)): flat.extend(p)
                else: flat.append(p)
            yield {"params": flat, "block": block_name, "frequency": frequency}

    params    = model.nora.cms.get_param_groups()
    optimizer = CMSAdamW(param_gen(params), lr=1e-4)
    check("optimizer created", optimizer is not None)

    # ── Training steps ────────────────────────────────────────────
    print(f"\n--- 5. Training ({TRAIN_STEPS} steps, accum={ACCUM_STEPS}) ---")
    model.train()
    losses      = []
    global_step = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        if step >= TRAIN_STEPS:
            break

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss / ACCUM_STEPS

        check(f"step {step+1} loss is not nan", not torch.isnan(loss))
        check(f"step {step+1} loss is not inf", not torch.isinf(loss))

        loss.backward()
        losses.append(loss.item() * ACCUM_STEPS)

        if (step + 1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("smoke/train_loss", loss.item() * ACCUM_STEPS, global_step)
            global_step += 1

        print(f"  step {step+1:02d}/{TRAIN_STEPS} | loss {loss.item()*ACCUM_STEPS:.4f}")

    check("all training losses finite", all(torch.isfinite(torch.tensor(l)) for l in losses))

    # ── Validation steps ──────────────────────────────────────────
    print(f"\n--- 6. Validation ({VAL_STEPS} steps) ---")
    model.eval()
    val_losses = []

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if step >= VAL_STEPS:
                break
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_losses.append(outputs.loss.item())
            print(f"  val step {step+1:02d}/{VAL_STEPS} | loss {outputs.loss.item():.4f}")

    avg_val = sum(val_losses) / len(val_losses)
    val_ppl = torch.exp(torch.tensor(avg_val)).item()
    writer.add_scalar("smoke/val_loss", avg_val, 0)
    check("all val losses finite", all(torch.isfinite(torch.tensor(l)) for l in val_losses))
    print(f"  avg val loss: {avg_val:.4f} | PPL: {val_ppl:.2f}")

    # ── Generation log ────────────────────────────────────────────
    print("\n--- 7. Generation log ---")
    prompts = ["The capital of France is", "Once upon a time"]
    model.eval()
    gen_results = []

    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **enc,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.0,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids      = out[0][enc["input_ids"].shape[-1]:]
            continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
            gen_results.append({"prompt": prompt, "continuation": continuation})
            print(f"  [{prompt}] → {continuation[:60]!r}")

    md_path = os.path.join(gen_dir, "smoke_gen.md")
    with open(md_path, "w") as f:
        for r in gen_results:
            f.write(f"**{r['prompt']}**\n\n{r['continuation']}\n\n---\n\n")

    check("gen log written to disk", os.path.exists(md_path))
    writer.add_text("smoke/generations", gen_results[0]["continuation"], 0)

    # ── Checkpoint save & reload ──────────────────────────────────
    print("\n--- 8. Checkpoint save & reload ---")
    ckpt_path = os.path.join(ckpt_dir, "smoke_checkpoint.pt")
    torch.save({
        "epoch":                1,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss":           sum(losses) / len(losses),
        "val_loss":             avg_val,
    }, ckpt_path)
    check("checkpoint file exists", os.path.exists(ckpt_path))
    check("checkpoint non-zero size", os.path.getsize(ckpt_path) > 0)

    # 1. Grab the original param for comparison BEFORE deleting the model.
    #    Use .clone() to ensure it persists in RAM.
    first_param_orig = next(model.parameters()).detach().cpu().clone()

    # 2. Free up VRAM!
    del model
    del optimizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Reload and verify weights match. 
    # Tip: Load to CPU first to avoid a VRAM spike during state_dict unpacking.
    ckpt   = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model2 = AutoModelForCausalLM.from_config(config)
    
    # Load weights into CPU model, then move the whole thing to GPU
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.to(device)

    first_param_reloaded = next(model2.parameters()).detach().cpu()
    check("reloaded weights match original", torch.allclose(first_param_orig, first_param_reloaded))

    # ── TensorBoard flush ─────────────────────────────────────────
    writer.flush()
    writer.close()
    check("tensorboard log dir exists", os.path.isdir(log_dir))

    # ── Memory report ─────────────────────────────────────────────
    if device.type == "cuda":
        used_gb  = torch.cuda.max_memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  Peak VRAM used : {used_gb:.2f} GB / {total_gb:.1f} GB")
        check("peak VRAM under 75% capacity", used_gb < total_gb * 0.75)

    # ── Done ──────────────────────────────────────────────────────
    print("\n" + "="*40)
    print("  ALL CHECKS PASSED — safe to train overnight")
    print("="*40 + "\n")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()