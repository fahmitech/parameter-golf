# 8192 BPE + Recurrent Factored Embedding + Parallel Muon

Moonshot submission scaffold derived from the `8192` factored-embedding branch, with shared-depth recurrence added on top of the March 23 banked training stack.

This is the high-upside path in the repo right now:

- `sp8192` tokenizer and dataset
- factored tied embedding to keep the larger vocabulary affordable
- banked transformer trunk with Parallel Muon
- shared-depth recurrence: a small set of unique blocks reused across multiple loops
- loop-conditioned adapters and recurrent skip weights
- existing mixed int6 export, sliding eval, and legal score-first TTT path kept compatible

## Status

This folder is runnable, but not benchmarked.

- `val_bpb`: pending
- artifact size: pending
- statistical significance: not established

## Architectural Delta

The prior `8192` scaffold only changed the embedding stack. This variant also changes the model family:

- `NUM_UNIQUE_BLOCKS` controls stored depth
- `RECURRENT_LOOPS` controls effective depth at runtime
- the same banked attention/MLP weights are reused across loops
- each reused block gets a small `LoopAdapter`
- recurrent skip weights are learned per loop

The default shape is intentionally conservative enough to remain plausible under the 10-minute budget:

- `NUM_UNIQUE_BLOCKS=4`
- `RECURRENT_LOOPS=3`
- effective depth: `12` block passes
- `MODEL_DIM=640`
- `NUM_HEADS=10`
- `NUM_KV_HEADS=5`
- `VOCAB_SIZE=8192`
- `EMBED_DIM=256`
- `XSA_LAST_N=2`
- `VE_LAYERS=auto` resolves to the deepest two unique blocks

## Why This Is the Main Moonshot

Quantization-only refinements are not enough to drive a `1.1194` model toward `1.0`. This branch attacks a higher-leverage bottleneck:

- larger tokenizer capacity
- more effective depth per stored parameter
- a cleaner path to later low-bit / recurrent / adaptation experiments

If anything in this repo is going to move materially toward `1.0`, it is more likely to be a compressed recurrent `8192`-token model than another small post-training quantization tweak on the `sp1024` stack.

## Recommended First Run

Start by measuring the base model without TTT. If the pre-TTT sliding score is not promising, TTT will not rescue it.

This is intentionally not the supervisor's `4x6x640` target yet. `4x6x640` is roughly `3.4x` the March-23 baseline's attention/MLP work, which is too aggressive for a first 600-second run. The safer ladder is `4x3x640`, then `4x4x640`, and only then `4x5/6x640` if step time remains viable.

```bash
RUN_ID=8192_recurrent_base \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 EMBED_DIM=256 \
NUM_UNIQUE_BLOCKS=4 RECURRENT_LOOPS=3 LOOP_ADAPTER_RANK=8 \
MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=3 \
BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=2 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=auto \
TTT_ENABLED=0 \
COMPILE_FULLGRAPH=0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.02 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If that looks competitive, rerun the same stack with `TTT_ENABLED=1`.

## Immediate Ablations

1. `NUM_UNIQUE_BLOCKS, RECURRENT_LOOPS`: try `4x3`, `4x4`, then `4x5` only if wallclock still fits.
2. `MODEL_DIM`: try `640`, `704`; do not jump to `768` until compile and step time are measured.
3. `EMBED_DIM`: try `0`, `256`, `320`.
4. `XSA_LAST_N`: try `0`, `2`.
5. `VE_LAYERS`: leave `auto` or pin to the deepest unique blocks, e.g. `2,3` for 4 unique blocks.
6. legal TTT only after the base sliding score is measured.
