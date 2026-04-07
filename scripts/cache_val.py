"""
verify_cache.py
───────────────
Explicitly verifies the tokenized dataset cache for correctness.
Checks: shape, dtype, padding, token distribution, dirty tokens,
        chunk boundaries, and decodes samples for human inspection.

Run with:
    python -m scripts.verify_cache
"""

import os
import sys
from collections import Counter

import torch
import datasets
from transformers import AutoTokenizer
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG — match your training script exactly
# ─────────────────────────────────────────────
EXP_NAME    = "nora_experiment_v5"
BASE_DIR    = f"./{EXP_NAME}"
CACHE_PATH  = os.path.join(BASE_DIR, "tokenized_cache")
TOKENIZER   = "meta-llama/Meta-Llama-3-8B"
MAX_SEQ_LEN = 2048

# Tokens that should NOT appear in clean data
DIRTY_SURFACE = ["下", "Tags", "Title", "@", "#", "男", "�"]

# How many chunks to fully decode and print
N_DECODE_SAMPLES = 5

# How many chunks to scan for dirty tokens (set to -1 for full scan)
N_DIRTY_SCAN = 500


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def check(label: str, condition: bool, detail: str = ""):
    status = "  OK  " if condition else "  FAIL"
    msg = f"[{status}] {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    if not condition:
        # Don't exit — collect all failures
        global _failures
        _failures.append(label)


_failures = []


def hr():
    print("─" * 70)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n=== CACHE VERIFICATION ===\n")

    # ── 0. Cache exists ───────────────────────────────────────────
    hr()
    print("0. Cache presence")
    hr()
    check("Cache directory exists", os.path.exists(CACHE_PATH), CACHE_PATH)
    train_path = os.path.join(CACHE_PATH, "train")
    val_path   = os.path.join(CACHE_PATH, "validation")
    check("train split exists",      os.path.exists(train_path))
    check("validation split exists", os.path.exists(val_path))

    if not os.path.exists(train_path):
        print("\nCache not found — nothing to verify. Run training once to build it.")
        sys.exit(1)

    # ── 1. Load ───────────────────────────────────────────────────
    hr()
    print("1. Loading splits")
    hr()
    train_ds = datasets.load_from_disk(train_path)
    val_ds   = datasets.load_from_disk(val_path)
    print(f"  Train chunks : {len(train_ds):,}")
    print(f"  Val chunks   : {len(val_ds):,}")

    # Expected range for wikitext-103 after filtering
    check("Train size reasonable (30k–60k)",
          30_000 <= len(train_ds) <= 60_000,
          f"got {len(train_ds):,}")
    check("Val size reasonable (500–3000)",
          500 <= len(val_ds) <= 3_000,
          f"got {len(val_ds):,}")

    # ── 2. Column names ───────────────────────────────────────────
    hr()
    print("2. Columns")
    hr()
    expected_cols = {"input_ids", "attention_mask", "labels"}
    train_cols    = set(train_ds.column_names)
    check("All required columns present",
          expected_cols.issubset(train_cols),
          f"found {train_cols}")

    # ── 3. Shape & dtype ──────────────────────────────────────────
    hr()
    print("3. Shape and dtype")
    hr()
    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    sample = train_ds[0]
    for col in ["input_ids", "attention_mask", "labels"]:
        if col not in sample:
            check(f"{col} shape", False, "column missing")
            continue
        t = sample[col]
        check(f"{col} shape == [{MAX_SEQ_LEN}]",
              list(t.shape) == [MAX_SEQ_LEN],
              f"got {list(t.shape)}")
        check(f"{col} dtype is torch.long",
              t.dtype == torch.long,
              f"got {t.dtype}")

    # ── 4. Padding check ──────────────────────────────────────────
    hr()
    print("4. Padding (should be zero — packed dataset has no padding)")
    hr()
    n_pad_tokens = 0
    n_scan       = min(200, len(train_ds))
    for i in range(n_scan):
        mask = train_ds[i]["attention_mask"]
        n_pad_tokens += (mask == 0).sum().item()

    check("Zero padding tokens in first 200 chunks",
          n_pad_tokens == 0,
          f"found {n_pad_tokens} padding tokens")

    # ── 5. Labels == input_ids ────────────────────────────────────
    hr()
    print("5. Labels == input_ids (CLM, no -100 masking needed)")
    hr()
    mismatches = 0
    for i in range(min(100, len(train_ds))):
        s = train_ds[i]
        if not torch.equal(s["input_ids"], s["labels"]):
            mismatches += 1
    check("labels == input_ids for first 100 chunks",
          mismatches == 0,
          f"{mismatches} mismatches found")

    # ── 6. Token ID range ─────────────────────────────────────────
    hr()
    print("6. Token ID range")
    hr()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size:,}")

    max_id = 0
    min_id = vocab_size
    for i in range(min(200, len(train_ds))):
        ids = train_ds[i]["input_ids"]
        max_id = max(max_id, ids.max().item())
        min_id = min(min_id, ids.min().item())

    check("All token IDs >= 0",         min_id >= 0,         f"min id = {min_id}")
    check("All token IDs < vocab_size", max_id < vocab_size, f"max id = {max_id}")
    print(f"  Token ID range in sample: [{min_id}, {max_id}]")

    # ── 7. Dirty token scan ───────────────────────────────────────
    hr()
    print(f"7. Dirty token scan (first {N_DIRTY_SCAN} chunks)")
    hr()
    dirty_ids = set()
    for surface in DIRTY_SURFACE:
        ids = tokenizer.encode(surface, add_special_tokens=False)
        dirty_ids.update(ids)
    print(f"  Dirty token IDs to check: {dirty_ids}")

    dirty_hits   = Counter()
    n_scan_dirty = min(N_DIRTY_SCAN, len(train_ds))
    for i in tqdm(range(n_scan_dirty), desc="  scanning"):
        ids = train_ds[i]["input_ids"].tolist()
        for tid in ids:
            if tid in dirty_ids:
                dirty_hits[tid] += 1

    if dirty_hits:
        print(f"  Dirty token counts:")
        for tid, count in dirty_hits.most_common():
            surface = tokenizer.decode([tid])
            print(f"    id={tid} '{surface}': {count} occurrences")
        check("No dirty tokens found", False,
              f"{sum(dirty_hits.values())} dirty tokens in {n_scan_dirty} chunks")
    else:
        check("No dirty tokens found", True,
              f"scanned {n_scan_dirty} chunks cleanly")

    # ── 8. EOS boundary check ─────────────────────────────────────
    hr()
    print("8. EOS token boundaries")
    hr()
    eos_id         = tokenizer.eos_token_id
    chunks_with_eos = 0
    n_eos_check    = min(200, len(train_ds))
    for i in range(n_eos_check):
        ids = train_ds[i]["input_ids"].tolist()
        if eos_id in ids:
            chunks_with_eos += 1

    eos_pct = chunks_with_eos / n_eos_check * 100
    print(f"  Chunks containing EOS: {chunks_with_eos}/{n_eos_check} ({eos_pct:.1f}%)")
    check("EOS tokens present (document boundaries)",
          chunks_with_eos > 0,
          "no EOS found — packing may not have inserted boundaries")

    # ── 9. Token frequency distribution ──────────────────────────
    hr()
    print("9. Token frequency — top 20 tokens in first 50 chunks")
    hr()
    freq = Counter()
    for i in range(min(50, len(train_ds))):
        freq.update(train_ds[i]["input_ids"].tolist())

    print(f"  {'Token ID':<12} {'Surface':<20} {'Count':<10}")
    print(f"  {'-'*8:<12} {'-'*18:<20} {'-'*6:<10}")
    for tid, count in freq.most_common(20):
        try:
            surface = repr(tokenizer.decode([tid]))
        except Exception:
            surface = f"<id={tid}>"
        print(f"  {tid:<12} {surface:<20} {count:<10}")

    # Flag if top token is suspiciously dominant
    top_count  = freq.most_common(1)[0][1]
    total_toks = sum(freq.values())
    top_pct    = top_count / total_toks * 100
    check("Top token < 15% of all tokens (no degenerate distribution)",
          top_pct < 15,
          f"top token is {top_pct:.1f}% of tokens")

    # ── 10. Human-readable decode ─────────────────────────────────
    hr()
    print(f"10. Decoded samples (first {N_DECODE_SAMPLES} chunks)")
    hr()
    for i in range(min(N_DECODE_SAMPLES, len(train_ds))):
        ids  = train_ds[i]["input_ids"].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        print(f"\n--- Chunk {i} (first 300 chars) ---")
        print(text[:300])

    # ── 11. Validation split sanity ───────────────────────────────
    hr()
    print("11. Validation split")
    hr()
    val_sample = val_ds[0]
    check("Val input_ids shape correct",
          list(val_sample["input_ids"].shape) == [MAX_SEQ_LEN],
          f"got {list(val_sample['input_ids'].shape)}")
    check("Val has no padding",
          val_sample["attention_mask"].min().item() == 1)

    # ── Summary ───────────────────────────────────────────────────
    hr()
    if _failures:
        print(f"RESULT: {len(_failures)} check(s) FAILED:")
        for f in _failures:
            print(f"  - {f}")
        print("\nDelete the cache and rebuild:")
        print(f"  rm -rf {CACHE_PATH}")
    else:
        print("RESULT: ALL CHECKS PASSED — cache is clean and ready")
    hr()
    print()


if __name__ == "__main__":
    main()