import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"]     = "expandable_segments:True"

import signal
import multiprocessing

import torch
import torch.nn as nn
import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Custom Imports
from model.CausalLM import NoraCausalLM
from config.CustomCLMConfig import NoraConfig
from config.ModelSettings import CMSConfig
from optimizers.AdamW import CMSAdamW


# ─────────────────────────────────────────────
# GENERATION HELPERS
# ─────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    prompts,
    device,
    max_new_tokens: int = 60,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.5,
) -> list[dict]:
    model.eval()
    results = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = out[0][enc["input_ids"].shape[-1]:]
        continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
        results.append({"prompt": prompt, "continuation": continuation})
    return results


def log_generations(
    results: list[dict],
    epoch: int,
    global_step: int,
    gen_log_dir: str,
    writer: SummaryWriter,
) -> None:
    md_path = os.path.join(gen_log_dir, f"epoch_{epoch:04d}_step_{global_step}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Generation log — Epoch {epoch} | Step {global_step}\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"## Sample {i}\n")
            f.write(f"**Prompt:** {r['prompt']}\n\n")
            f.write(f"**Continuation:**\n\n{r['continuation']}\n\n")
            f.write("---\n\n")
    print(f"  [gen_log] saved → {md_path}")

    tb_text = ""
    for r in results:
        tb_text += f"**{r['prompt']}**  \n{r['continuation']}\n\n---\n\n"
    writer.add_text("Generations/samples", tb_text, global_step=global_step)


def param_gen(param_dict):
    for block_name, (frequency, block_params) in param_dict.items():
        flat = []
        for p in block_params:
            if isinstance(p, (list, tuple)):
                flat.extend(p)
            else:
                flat.append(p)
        yield {"params": flat, "block": block_name, "frequency": frequency}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── 1. Directories ────────────────────────────────────────────
    EXP_NAME       = "nora_experiment_v5"
    BASE_DIR       = f"./{EXP_NAME}"
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR        = os.path.join(BASE_DIR, "logs")
    GEN_LOG_DIR    = os.path.join(BASE_DIR, "gen_logs")
    CACHE_PATH     = "./nora_experiment_v7/tokenized_cache"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GEN_LOG_DIR,    exist_ok=True)

    writer = SummaryWriter(log_dir=LOG_DIR)

    # ── Signal handlers ───────────────────────────────────────────
    def cleanup(sig, frame):
        print("\nInterrupted — cleaning up workers and closing writer …")
        for child in multiprocessing.active_children():
            child.terminate()
        writer.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # ── 2. Memory & performance flags ─────────────────────────────
    # TF32: free ~15 % speedup on H100 with negligible precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 3. Config & model ─────────────────────────────────────────
    # CMS Config rationale (Nested Learning):
    #   frequencies [1,2,4,8,16]  — strict frequency hierarchy; slow→fast nesting
    #                                mirrors multi-scale decomposition in TTT /
    #                                Mamba-style models. Low-frequency blocks update
    #                                rarely (global semantics), high-frequency blocks
    #                                update often (local token patterns).
    #   expansions  [4,4,3,2,2]   — larger MLP capacity at slow/semantic blocks;
    #                                smaller at fast blocks so TTT online adaptation
    #                                stays stable and doesn't overfit on short windows.
    #   SiLU                       — native to LLaMA-3; smoother gradients than GELU
    #                                for both pretraining and TTT gradient steps.
    cms_config = CMSConfig(
        3072,
        3,                    # was 7 — fewer blocks, less overfitting
        [1, 4, 16],           # was [1,2,4,8,16,32,64] — wider frequency gaps
        [4, 2, 1],            # was [6,4,4,3,3,2,2] — shrinking capacity
        nn.GELU,
    )
    config = NoraConfig(model_name="llama-3.2-3b", cms_cfg=cms_config)

    AutoConfig.register("NORA", NoraConfig)
    AutoModelForCausalLM.register(NoraConfig, NoraCausalLM)

    model = AutoModelForCausalLM.from_config(config)
    model = torch.compile(model)
    # ── Checkpoint resume ─────────────────────────────────────────
    START_EPOCH = 15
    checkpoint  = None

    if START_EPOCH > 0:
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, f"checkpoint_epoch_{START_EPOCH}.pt"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"START_EPOCH={START_EPOCH} but no checkpoint at {checkpoint_path}"
            )
        checkpoint = torch.load(
            checkpoint_path, weights_only=False, map_location="cuda"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {START_EPOCH}")
    else:
        print("No checkpoint — starting from scratch.")

    
    model.to(device)
    # for param in model.nora.lm_head.parameters():
    #     param.requires_grad = True
    # ── 4. Tokenizer ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    print(tokenizer.name_or_path)

    # Check if '下' is even in the vocabulary
    ids = tokenizer.encode("下", add_special_tokens=False)
    print(f"'下' encodes to: {ids}")
    print(f"Those IDs decode to: {[tokenizer.decode([i]) for i in ids]}")

    tokenizer.pad_token = tokenizer.eos_token

    MAX_SEQ_LEN = 2048

    
    def is_clean(text: str) -> bool:
        t = text.strip()
        if len(t) < 20:            return False  # blank lines, section markers
        if t.startswith("="):      return False  # = = Heading = =
        if t.startswith("<"):      return False  # <br>, html tags
        if "Tags" in t:            return False  # template artifacts
        if "\u4e0b" in t:          return False  # stray CJK unicode (下 etc.)
        if t.startswith("@"):      return False  # email/handle artifacts
        if t.count("\"") > 10:     return False  # lines that are mostly punctuation
        return True

    def pack_split(raw_split) -> datasets.Dataset:
        """
        Concatenate all clean document tokens into one stream,
        insert EOS at document boundaries, then slice into
        fixed MAX_SEQ_LEN chunks with no padding.
        """
        all_ids = []
        for example in tqdm(raw_split, desc="  packing"):
            if not is_clean(example["text"]):
                continue
            ids = tokenizer(
                example["text"],
                truncation=False,
                padding=False,
            )["input_ids"]
            if len(ids) == 0:
                continue
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)  # document boundary marker

        # Trim to exact multiple of MAX_SEQ_LEN — no partial chunks
        total   = (len(all_ids) // MAX_SEQ_LEN) * MAX_SEQ_LEN
        all_ids = all_ids[:total]
        from collections import Counter

        counter = Counter(all_ids)

        # decode top tokens
        top = counter.most_common(50)

        for tok_id, freq in top:
            print(tok_id, tokenizer.decode([tok_id]), freq)
        chunks = [
            all_ids[i : i + MAX_SEQ_LEN]
            for i in range(0, total, MAX_SEQ_LEN)
        ]

        print(f"  → {len(chunks):,} chunks from {len(all_ids):,} tokens")

        return datasets.Dataset.from_dict({
            "input_ids":      chunks,
            "attention_mask": [[1] * MAX_SEQ_LEN for _ in chunks],
            "labels":         chunks,   # CLM: labels == input_ids, no -100 needed
        })                              #      since there is zero padding

    if os.path.exists(CACHE_PATH):
        print("Loading packed dataset from cache …")
        tokenized_dataset = datasets.DatasetDict({
            "train":      datasets.load_from_disk(os.path.join(CACHE_PATH, "train")),
            "validation": datasets.load_from_disk(os.path.join(CACHE_PATH, "validation")),
        })
    else:
        print("Building packed dataset (one-time cost) …")
        raw_dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1")

        train_packed = pack_split(raw_dataset["train"])
        val_packed   = pack_split(raw_dataset["validation"])

        train_packed.save_to_disk(os.path.join(CACHE_PATH, "train"))
        val_packed.save_to_disk(os.path.join(CACHE_PATH, "validation"))
        print(f"Saved packed dataset → {CACHE_PATH}")

        tokenized_dataset = datasets.DatasetDict({
            "train":      train_packed,
            "validation": val_packed,
        })

    tokenized_dataset.set_format(type="torch")
    train_dataset = tokenized_dataset["train"]
    val_dataset   = tokenized_dataset["validation"]

    # Batch size 4 + ACCUM_STEPS 4 = effective batch 16 at ~½ the VRAM of batch 10
    BATCH_SIZE  = 8
    ACCUM_STEPS = 2     # effective batch size = BATCH_SIZE × ACCUM_STEPS = 16

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # ── 6. Generation prompts ─────────────────────────────────────
    EVAL_PROMPTS = [
        # Wikitext-style openers — tests if model learned encyclopedic prose structure
        "The history of the United States",
        "During the first World War",
        "The film was directed by",
        
        # Syntactic completion — tests grammar without requiring facts
        "The researchers concluded that",
        "Although the results were",
        
        # Keep one factual to track knowledge emergence over epochs
        "The capital of France is",
    ]

    # ── 7. Optimizer & scaler ─────────────────────────────────────
    params    = model.nora.cms.get_param_groups()
    optimizer = CMSAdamW(param_gen(params), lr=1e-4)

    if checkpoint is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # No GradScaler needed — it only works with FP16, not BF16.
    # BF16 has the same exponent range as FP32 so gradients never underflow.

    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # ── 8. Training loop ──────────────────────────────────────────
    num_epochs             = 20
    global_step            = 0
    GEN_LOG_EVERY_N_EPOCHS = 1

    for epoch in range(START_EPOCH, START_EPOCH + num_epochs):
        model.train()
        total_train_loss = 0.0
        train_batches    = 0

        optimizer.zero_grad()   # zero once before the accumulation window

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for step, batch in enumerate(train_bar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # BF16 autocast — halves activation memory on H100
            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                # Divide loss here so gradients are already averaged when
                # we call optimizer.step() after ACCUM_STEPS micro-steps.
                loss = outputs.loss / ACCUM_STEPS

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss.backward()

            # Only step + log every ACCUM_STEPS micro-batches
            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                # Unscale for logging (multiply back to get the true loss)
                true_loss = loss.item() * ACCUM_STEPS
                writer.add_scalar("Loss/train_step", true_loss, global_step)
                global_step += 1
                train_bar.set_postfix({"loss": f"{true_loss:.4f}"})

            total_train_loss += loss.item() * ACCUM_STEPS
            train_batches    += 1

        # Flush any leftover gradient if dataset size isn't divisible by ACCUM_STEPS
        if (len(train_loader)) % ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / max(train_batches, 1)

        # ── Validation ───────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch in val_bar:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                with autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # ── Perplexity ───────────────────────────────────────────
        train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
        val_ppl   = torch.exp(torch.tensor(avg_val_loss)).item()

        # ── TensorBoard scalars ──────────────────────────────────
        writer.add_scalar("Loss/train_epoch",       avg_train_loss, epoch)
        writer.add_scalar("Loss/val_epoch",         avg_val_loss,   epoch)
        writer.add_scalar("Perplexity/train_epoch", train_ppl,      epoch)
        writer.add_scalar("Perplexity/val_epoch",   val_ppl,        epoch)

        print(
            f"\nEnd of Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} (PPL {train_ppl:.2f}) | "
            f"Val Loss: {avg_val_loss:.4f} (PPL {val_ppl:.2f})\n"
        )

        # ── Qualitative generation log ───────────────────────────
        if (epoch + 1) % GEN_LOG_EVERY_N_EPOCHS == 0:
            print("  [gen_log] running generation samples …")
            gen_results = generate_samples(
                model, tokenizer, EVAL_PROMPTS, device
            )
            log_generations(
                gen_results,
                epoch=epoch + 1,
                global_step=global_step,
                gen_log_dir=GEN_LOG_DIR,
                writer=writer,
            )
            model.train()

        # ── Checkpoint ───────────────────────────────────────────
        torch.save(
            {
                "epoch":                epoch + START_EPOCH + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss":           avg_train_loss,
                "val_loss":             avg_val_loss,
            },
            os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pt"),
        )

    writer.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()