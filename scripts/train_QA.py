"""
train_lora_qa.py
────────────────
LoRA fine-tuning of NoraCausalLM on SQuAD v2 for grounded QA.

Key design decisions
────────────────────
1.  PROMPT FORMAT — Wikipedia-flavoured completion so the model stays
    in-distribution.  The answer is terminated by a special <|end|>
    sentinel token that we add to the vocabulary and embed, which
    directly fixes the "model doesn't know when to stop" problem.

2.  LOSS MASKING — cross-entropy is computed ONLY on answer + sentinel
    tokens.  Prompt / context tokens are masked to -100 so the model
    isn't rewarded for memorising the question.

3.  LORA TARGET — we inject LoRA only into the Llama attention
    projections (q_proj, v_proj).  The CMS blocks are frozen
    completely; only the LoRA adapters + the new sentinel embedding
    are trained.

4.  UNANSWERABLE — SQuAD v2 has ~33 % unanswerable questions.
    We map those to the literal answer string "unanswerable" so the
    model learns to say so rather than hallucinate.

Usage
─────
    pip install peft datasets tqdm
    python train_lora_qa.py
"""

import os, signal, multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"]     = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from tqdm import tqdm

# ── Your custom imports (adjust paths if needed) ──────────────────────────────
from model.CausalLM import NoraCausalLM
from config.CustomCLMConfig import NoraConfig
from config.ModelSettings import CMSConfig

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

EXP_NAME        = "nora_lora_qa_v1"
BASE_DIR        = f"./{EXP_NAME}"
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

# Pretrained NoraCausalLM checkpoint to start from
NORA_CHECKPOINT = "./nora_experiment_v5/checkpoints/checkpoint_epoch_15.pt"

# LoRA hyper-parameters
LORA_R          = 16        # rank — 8 or 16 is usually enough for QA
LORA_ALPHA      = 32        # scale = alpha/r; 2× r is a safe default
LORA_DROPOUT    = 0.05
# Only inject into attention projections; CMS blocks stay frozen
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Training hyper-parameters
LEARNING_RATE   = 2e-4      # higher than pre-training is fine for LoRA
BATCH_SIZE      = 4
ACCUM_STEPS     = 4         # effective batch = 16
NUM_EPOCHS      = 5
MAX_SEQ_LEN     = 512       # QA pairs are short; saves memory

# SQuAD v2 subset sizes (set to None to use the full split)
MAX_TRAIN       = 20_000
MAX_VAL         = 2_000

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia-flavoured so the model stays in its training distribution.
# <|end|> is the sentinel that teaches the model when to stop.

SENTINEL = "<|end|>"

PROMPT_TEMPLATE = (
    "The following is a passage from Wikipedia.\n\n"
    "{context}\n\n"
    "According to the passage above, {question_lower} "
)
# The *label* appended during training:  "{answer}{SENTINEL}"


def build_prompt(context: str, question: str) -> str:
    q = question.strip()
    # Lower-case first char so the sentence reads naturally as a completion
    q_lower = q[0].lower() + q[1:] if q else q
    # Strip trailing "?" — we want the model to complete, not answer a question
    q_lower = q_lower.rstrip("?").strip()
    return PROMPT_TEMPLATE.format(context=context.strip(), question_lower=q_lower)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SQuADQADataset(Dataset):
    """
    Converts SQuAD v2 rows into packed token sequences with loss masking.

    Full sequence:  [PROMPT tokens] [ANSWER tokens] [SENTINEL token]
    Label mask:     [-100 ... -100] [ANSWER tokens] [SENTINEL token]

    This means cross-entropy is only computed on the answer span.
    """

    def __init__(self, squad_split, tokenizer, max_seq_len: int, max_samples=None):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.sentinel_id = tokenizer.convert_tokens_to_ids(SENTINEL)
        self.items       = []

        data = squad_split if max_samples is None else squad_split.select(range(max_samples))

        for row in tqdm(data, desc="Tokenising SQuAD"):
            context  = row["context"]
            question = row["question"]
            answers  = row["answers"]["text"]

            # SQuAD v2: empty answers = unanswerable
            answer = answers[0] if answers else "unanswerable"

            prompt = build_prompt(context, question)

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_len - 10,   # leave room for answer + sentinel
            )["input_ids"]

            answer_ids = tokenizer(
                answer,
                add_special_tokens=False,
                truncation=True,
                max_length=40,
            )["input_ids"]

            # Full input:  prompt + answer + sentinel
            input_ids = prompt_ids + answer_ids + [self.sentinel_id]

            # Labels:  mask prompt, keep answer + sentinel
            labels = (
                [-100] * len(prompt_ids)
                + answer_ids
                + [self.sentinel_id]
            )

            # Pad / truncate to max_seq_len
            pad_len = max_seq_len - len(input_ids)
            if pad_len < 0:
                # sequence too long — truncate from the left of the context
                overflow = -pad_len
                input_ids = input_ids[overflow:]
                labels    = labels[overflow:]
                pad_len   = 0

            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids      = input_ids + [tokenizer.pad_token_id] * pad_len
            labels         = labels    + [-100]                   * pad_len

            self.items.append({
                "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels":         torch.tensor(labels,         dtype=torch.long),
            })

    def __len__(self):  return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION (for qualitative eval)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    context: str,
    question: str,
    device,
    max_new_tokens: int = 60,
    temperature: float  = 0.3,   # lower than pre-training: we want focused answers
    top_k: int          = 50,
    top_p: float        = 0.9,
    repetition_penalty: float = 1.3,
) -> str:
    sentinel_id = tokenizer.convert_tokens_to_ids(SENTINEL)
    prompt      = build_prompt(context, question)
    enc         = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids   = enc["input_ids"]

    for _ in range(max_new_tokens):
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids)

        logits = out.logits[0, -1, :].float()

        # Repetition penalty
        for prev_id in input_ids[0].tolist():
            logits[prev_id] = (
                logits[prev_id] / repetition_penalty
                if logits[prev_id] > 0
                else logits[prev_id] * repetition_penalty
            )

        logits /= temperature

        # Top-k
        filtered = torch.full_like(logits, float("-inf"))
        topk_v, topk_i = torch.topk(logits, top_k)
        filtered[topk_i] = topk_v

        # Top-p
        sorted_l, sorted_i = torch.sort(filtered, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
        sorted_l[cum - torch.softmax(sorted_l, dim=-1) > top_p] = float("-inf")
        filtered[sorted_i] = sorted_l

        next_id = torch.multinomial(torch.softmax(filtered, dim=-1), 1)
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=-1)

        # Stop on sentinel OR native EOS
        if next_id.item() in (sentinel_id, tokenizer.eos_token_id):
            break

    new_ids = input_ids[0][enc["input_ids"].shape[-1]:]
    answer  = tokenizer.decode(new_ids, skip_special_tokens=False)
    # Strip sentinel and everything after
    answer  = answer.split(SENTINEL)[0].strip()
    return answer


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)

    def cleanup(sig, frame):
        print("\nInterrupted — shutting down …")
        for child in multiprocessing.active_children():
            child.terminate()
        writer.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Tokenizer + sentinel token ─────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token

    # Add the stopping sentinel — this is what cures the "doesn't stop" problem.
    # The embedding for the new token is initialised to the mean of all existing
    # embeddings, which is a safe warm-start (better than random).
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": [SENTINEL]})
    print(f"Added {num_added} new token(s): {SENTINEL!r}  "
          f"(id={tokenizer.convert_tokens_to_ids(SENTINEL)})")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    cms_config = CMSConfig(
        3072,
        7,
        [1, 2, 4, 8, 16, 32, 64],
        [6, 4, 4, 3, 3, 2, 2],
        nn.GELU,
    )
    config = NoraConfig(model_name="llama-3.2-3b", cms_cfg=cms_config)

    AutoConfig.register("NORA", NoraConfig)
    AutoModelForCausalLM.register(NoraConfig, NoraCausalLM)

    model = AutoModelForCausalLM.from_config(config)

    # Load your pretrained weights
    checkpoint = torch.load(NORA_CHECKPOINT, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded NoraCausalLM weights from {NORA_CHECKPOINT}")

    # Resize embeddings to accommodate the new sentinel token.
    # New row is initialised to the mean of existing rows (warm-start).
    old_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_vocab = model.get_input_embeddings().weight.shape[0]
    if new_vocab > old_vocab:
        with torch.no_grad():
            mean_embed = model.get_input_embeddings().weight[:old_vocab].mean(0)
            model.get_input_embeddings().weight[old_vocab:] = mean_embed
            # Mirror into the LM head if it's tied or separate
            lm_head = model.get_output_embeddings()
            if lm_head is not None and lm_head.weight.shape[0] > old_vocab:
                lm_head.weight[old_vocab:] = mean_embed
        print(f"Embedding table extended: {old_vocab} → {new_vocab}")

    # ── 3. LoRA wrapping ──────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type        = TaskType.CAUSAL_LM,
        r                = LORA_R,
        lora_alpha       = LORA_ALPHA,
        lora_dropout     = LORA_DROPOUT,
        target_modules   = LORA_TARGET_MODULES,
        # Bias: "none" keeps parameter count minimal
        bias             = "none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # Expected output: ~0.3–0.5 % of total params are trainable (LoRA + sentinel)

    model.to(device)

    # ── 4. Dataset ────────────────────────────────────────────────────────────
    print("Loading SQuAD v2 …")
    squad = load_dataset("rajpurkar/squad_v2")

    train_ds = SQuADQADataset(
        squad["train"],      tokenizer, MAX_SEQ_LEN, max_samples=MAX_TRAIN
    )
    val_ds   = SQuADQADataset(
        squad["validation"], tokenizer, MAX_SEQ_LEN, max_samples=MAX_VAL
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )

    # ── 5. Optimizer ──────────────────────────────────────────────────────────
    # Only LoRA params + the new sentinel embedding row are trainable.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # Cosine LR schedule over all training steps
    total_steps = (len(train_loader) // ACCUM_STEPS) * NUM_EPOCHS
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LEARNING_RATE * 0.1
    )

    # ── 6. Qualitative eval examples ──────────────────────────────────────────
    EVAL_EXAMPLES = [
        {
            "context":  "The Eiffel Tower is located in Paris, France. "
                        "It was constructed between 1887 and 1889 as the entrance arch "
                        "for the 1889 World's Fair.",
            "question": "Where is the Eiffel Tower located?",
        },
        {
            "context":  "Photosynthesis is a process used by plants to convert light "
                        "energy into chemical energy stored in glucose.",
            "question": "What do plants produce during photosynthesis?",
        },
        {
            "context":  "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
            "question": "What country invented the internet?",   # unanswerable
        },
    ]

    # ── 7. Training loop ──────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        train_batches    = 0
        optimizer.zero_grad()

        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        for step, batch in enumerate(bar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss / ACCUM_STEPS

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss.backward()

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                true_loss = loss.item() * ACCUM_STEPS
                writer.add_scalar("Loss/train_step", true_loss, global_step)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
                global_step += 1
                bar.set_postfix({"loss": f"{true_loss:.4f}"})

            total_train_loss += loss.item() * ACCUM_STEPS
            train_batches    += 1

        # Flush leftover gradients
        if len(train_loader) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / max(train_batches, 1)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += out.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        train_ppl    = torch.exp(torch.tensor(avg_train_loss)).item()
        val_ppl      = torch.exp(torch.tensor(avg_val_loss)).item()

        writer.add_scalar("Loss/train_epoch",       avg_train_loss, epoch)
        writer.add_scalar("Loss/val_epoch",         avg_val_loss,   epoch)
        writer.add_scalar("Perplexity/train_epoch", train_ppl,      epoch)
        writer.add_scalar("Perplexity/val_epoch",   val_ppl,        epoch)

        print(
            f"\nEpoch {epoch} | "
            f"Train Loss {avg_train_loss:.4f} (PPL {train_ppl:.2f}) | "
            f"Val Loss {avg_val_loss:.4f} (PPL {val_ppl:.2f})"
        )

        # ── Qualitative samples ───────────────────────────────────────────────
        print("\n── Generation samples ──")
        for ex in EVAL_EXAMPLES:
            ans = generate_answer(model, tokenizer, ex["context"], ex["question"], device)
            print(f"  Q: {ex['question']}")
            print(f"  A: {ans}\n")

        # ── Save LoRA adapter only (small — ~few MB) ──────────────────────────
        adapter_path = os.path.join(CHECKPOINT_DIR, f"lora_adapter_epoch_{epoch}")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"Saved LoRA adapter → {adapter_path}\n")

    writer.close()
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPER (import this in your RAG pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def load_for_inference(adapter_path: str, base_checkpoint: str, device="cuda"):
    """
    Re-loads the base NoraCausalLM and merges the LoRA adapter for inference.

    Usage:
        model, tokenizer = load_for_inference(
            adapter_path   = "./nora_lora_qa_v1/checkpoints/lora_adapter_epoch_5",
            base_checkpoint= "./nora_experiment_v5/checkpoints/checkpoint_epoch_15.pt",
        )
        answer = generate_answer(model, tokenizer, context, question, device)
    """
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    cms_config = CMSConfig(
        3072, 7,
        [1, 2, 4, 8, 16, 32, 64],
        [6, 4, 4, 3, 3, 2, 2],
        nn.GELU,
    )
    config = NoraConfig(model_name="llama-3.2-3b", cms_cfg=cms_config)
    AutoConfig.register("NORA", NoraConfig)
    AutoModelForCausalLM.register(NoraConfig, NoraCausalLM)

    base = AutoModelForCausalLM.from_config(config)
    ckpt = torch.load(base_checkpoint, weights_only=False, map_location="cpu")
    base.load_state_dict(ckpt["model_state_dict"])
    base.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()   # fuses LoRA weights → no inference overhead
    model.to(device).eval()

    return model, tokenizer


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()