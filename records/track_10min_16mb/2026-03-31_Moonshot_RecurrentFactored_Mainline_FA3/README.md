# Moonshot Recurrent-Factored Mainline (FA3)

Runnable moonshot scaffold that pushes beyond the March 25 public SOTA by changing the
compute-vs-parameters tradeoff instead of just polishing export.

This branch keeps the proven banked training/export stack, but replaces the 11 unique
block trunk with a shared-depth recurrent transformer:

- `5` unique blocks
- `3` recurrent loops
- effective depth: `15` block passes
- factorized tied embedding (`EMBED_DIM=256`, `MODEL_DIM=640`)
- BigramHash `3072 x 112`
- XSA on all unique blocks
- FlashAttention-3 runtime path
- no TTT in the base recipe

The defaults are intentionally set to the runnable `sp1024` data path first, so we can
benchmark the architecture now. If the curve is good, the same model family can later be
pushed toward a larger tokenizer.

## Status

- `val_bpb`: pending
- artifact bytes: pending
- verification: `python3 -m py_compile train_gpt.py`

## Why This Is Different

The public SOTA stores 11 separate blocks. This moonshot stores only 5 and reuses them
across 3 loops with:

- per-block loop adapters
- learned recurrent skip weights
- learned loop gates

The hypothesis is that, under a strict 16 MB artifact cap, deeper reused computation is a
better way to spend bits than another small quantization refinement.

## Default Shape

- `VOCAB_SIZE=1024`
- `NUM_UNIQUE_BLOCKS=5`
- `RECURRENT_LOOPS=3`
- `MODEL_DIM=640`
- `NUM_HEADS=10`
- `NUM_KV_HEADS=5`
- `EMBED_DIM=256`
- `BIGRAM_VOCAB_SIZE=3072`
- `BIGRAM_DIM=112`
- `XSA_LAST_N=5`
- `WARMDOWN_ITERS=4000`
- `TTT_ENABLED=0`

## Requirements

- `flash_attn_interface` / FlashAttention-3 available in the environment
- `sentencepiece`
- `zstandard`

## Recommended 1x H100 Sanity Run

```bash
OMP_NUM_THREADS=1 \
RUN_ID=moonshot_recurrent_2000_seed1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
EMBED_DIM=256 \
NUM_UNIQUE_BLOCKS=5 RECURRENT_LOOPS=3 LOOP_ADAPTER_RANK=8 \
MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=3 \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 XSA_LAST_N=5 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=auto \
TTT_ENABLED=0 COMPILE_FULLGRAPH=0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=1200 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
ITERATIONS=2000 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Recommended 8x H100 Target Run

```bash
RUN_ID=moonshot_recurrent_seed314 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
EMBED_DIM=256 \
NUM_UNIQUE_BLOCKS=5 RECURRENT_LOOPS=3 LOOP_ADAPTER_RANK=8 \
MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=3 \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 XSA_LAST_N=5 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=auto \
TTT_ENABLED=0 COMPILE_FULLGRAPH=0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=4000 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
ITERATIONS=7000 MAX_WALLCLOCK_SECONDS=600 \
SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Read This Before Scaling Up

- If the `2000`-step sanity run is not clearly ahead of the plain `sp1024` baselines by
  validation BPB, do not spend for the full run.
- If this trunk looks alive, the next ladder is `RECURRENT_LOOPS=4` before touching TTT.
- If `sp8192` becomes available cleanly, this is the first architecture to port.
