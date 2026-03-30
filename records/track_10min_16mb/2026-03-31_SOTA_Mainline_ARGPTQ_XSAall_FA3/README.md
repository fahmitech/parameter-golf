# SOTA Mainline AR-GPTQ + XSA-all + BigramHash 3072x112 (FA3)

Runnable derivative scaffold of the March 25, 2026 public SOTA record:
`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`.

This folder is meant to be a strong mainline for new experiments on RunPod. It keeps
the March 25 trunk and export recipe, including the direct
`flash_attn_interface` / FlashAttention-3 runtime path.

## Status

- `val_bpb`: pending
- artifact bytes: pending
- verification: `python3 -m py_compile train_gpt.py`

## What This Keeps

- AR self-generated Full Hessian GPTQ calibration
- XSA on all 11 layers
- BigramHash `3072 x 112`
- Parallel Muon + parameter banking
- selective `+-1` pruning
- `lzma preset=9`
- no TTT in the mainline recipe

## What Changed Here

- defaults now match the March 25 winning recipe:
  - `BIGRAM_VOCAB_SIZE=3072`
  - `BIGRAM_DIM=112`
  - `WARMDOWN_ITERS=4000`
- metadata is reset to pending so this folder does not claim the original record's scores

## Requirements

- `flash_attn_interface` / FlashAttention-3 available in the environment
- `sentencepiece`
- `zstandard`

## Suggested Full Run

```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Suggested 1x H100 Sanity Run

```bash
OMP_NUM_THREADS=1 \
RUN_ID=sota_mainline_2000_seed1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=1200 \
ITERATIONS=2000 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Notes

- This is a candidate mainline experiment, not a claimed reproduction of the March 25 result.
- If this branch proves healthy, the next deltas should be small and explicit so we can
  measure whether they actually beat the public SOTA recipe.
