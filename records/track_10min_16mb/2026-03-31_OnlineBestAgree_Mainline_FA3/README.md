# Online Best-Agree Mainline (FA3)

Runnable derivative scaffold of the current strongest open-PR line:

- [PR #1060](https://github.com/openai/parameter-golf/pull/1060): Loader + Full GPTQ + XSA-all
- [PR #1145](https://github.com/openai/parameter-golf/pull/1145): online token/within-word/word-start agreement overlay

This folder is intended as the new mainline for serious SOTA attempts. It keeps the
`#1145` training/eval code path intact, including the native `online_ngram_state.c`
helper and the causal normalized online agreement evaluator in
`online_best_agree_eval.py`.

## Status

- `val_bpb`: pending
- artifact bytes: pending
- verification:
  - `python3 -m py_compile train_gpt.py`
  - `python3 -m py_compile online_best_agree_eval.py`

## Why This Is The Mainline

Relative to the March 25 public leader, this family has the strongest current frontier
signal:

- stronger base training stack than the March 23 line
- Full Hessian GPTQ
- XSA on all 11 layers
- pure sliding base evaluation, no TTT
- a legal prefix-only online agreement overlay that appears to add about `0.003` BPB

This is the highest-probability path right now because it changes the minimum number of
things while building on the newest promising results.

## Core Idea

At eval time, the online overlay keeps three causal experts derived only from the strict
prefix:

- token n-gram top-token hints
- within-word continuation hints
- word-start first-token hints

At each scored position it selects at most one hinted token, optionally adds a small
agreement bonus when multiple experts support the same token, and applies a single-token
boost inside the model's already-normalized distribution.

## Suggested 1x H100 Sanity Run

```bash
OMP_NUM_THREADS=1 \
RUN_ID=online_agree_2000_seed1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112 XSA_LAST_N=11 \
USE_GPTQ=1 TTT_ENABLED=0 ONLINE_BEST_AGREE_EVAL=1 EVAL_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=0 GPTQ_RESERVE_MS=10000 \
WARMDOWN_ITERS=1200 TIED_EMBED_LR=0.035 ITERATIONS=2000 TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Suggested 8x H100 Target Run

```bash
SEED=1337 \
BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112 XSA_LAST_N=11 \
USE_GPTQ=1 TTT_ENABLED=0 ONLINE_BEST_AGREE_EVAL=1 EVAL_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=600 GPTQ_RESERVE_MS=10000 \
WARMDOWN_ITERS=4000 TIED_EMBED_LR=0.035 ITERATIONS=6700 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Notes

- This folder is a derivative scaffold for testing on your fork, not a claim of the
  original PR's measured result.
- If this line is healthy on your setup, the next improvement should stay inside this
  family: better online agreement experts or a slightly stronger trunk underneath them.
