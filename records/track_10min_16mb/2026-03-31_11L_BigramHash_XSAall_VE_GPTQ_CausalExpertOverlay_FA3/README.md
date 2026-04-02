# Record: 11L BigramHash + XSA-all + VE + Full GPTQ + Causal Expert Overlay (FA3)

**Status:** mainline / experimental scaffold, not a verified leaderboard submission yet.

**Preserved local baseline result:** `1.1984 val_bpb` at end of a 2000-step 1x H100 sanity run | `1.20938784` int6 roundtrip BPB | `14,156,425` bytes total.

**Current code in this folder now also includes an experimental in-model expert mixer,** but that addition has **not** been measured yet. The preserved result below predates that change.

**Important:** the preserved March 31 local result below predates the current legality fix. The current `train_gpt.py` still calibrates GPTQ Hessians from training batches, but now does so **inside the counted training budget and before any post-training evaluation**, so the eval phase no longer re-accesses training data.

---

## Preserved Local Result

The latest surviving local RunPod snippet is saved in [`LOCAL_RESULT_2026-03-31.md`](./LOCAL_RESULT_2026-03-31.md). It should be treated as a manually preserved baseline result, not as complete submission evidence.

| Run | GPUs | Steps | End-of-train BPB | Post-EMA BPB | Int6 roundtrip BPB | Total bytes |
|-----|------|-------|------------------|--------------|--------------------|-------------|
| Preserved local sanity run | 1x H100 | 2000 | `1.1984` | `1.2039` | `1.20938784` | `14,156,425` |

Additional preserved numbers from the snippet:

- GPTQ calibrated layers: `66`
- Peak allocated GPU memory: `23,003 MiB`
- GPTQ calibration time: `6.7s`
- Serialized int6 artifact: `14,011,228` bytes

The surviving snippet does **not** include:

- `final_int6_sliding_window_exact`
- `online_best_agree:done`
- `online_best_agree_sliding_window`
- the full run header / exact command used on RunPod

---

## Main Idea

This folder is a runnable hybrid mainline built from:

- [PR #1060](https://github.com/openai/parameter-golf/pull/1060): loader + Full Hessian GPTQ + XSA-all
- [PR #1145](https://github.com/openai/parameter-golf/pull/1145): online causal agreement overlay

The base thesis is:

1. Keep a strong compact base model under the 16 MB constraint.
2. Use Full Hessian GPTQ and mixed critical-layer precision to preserve as much quality as possible after export.
3. Recover additional compression signal from strictly causal prefix experts at eval time.

This record now pushes that one step further with a small **in-model causal expert mixer** so some of the structure that previously lived only in the eval overlay can be learned directly by the model.

---

## Main Changes In This Folder

### 1. Full GPTQ + XSA-all mainline

Relative to older March public lines, this folder standardizes on:

- `BigramHash = 3072 x 112`
- `XSA_LAST_N = 11`
- `ROPE_DIMS = 16`
- `VE_LAYERS = 7,8,9,10`
- optional `ATTENTION_SINK = 1`
- Full Hessian GPTQ int6 with `GPTQ_BLOCK_SIZE = 64`
- mixed critical-layer precision via `INT8_ATTN_LAYERS = 0,-1`

This is the strongest compact-model base stack currently represented in this record folder.

### 2. Online Best-Agree overlay

The eval path keeps a prefix-only agreement overlay that combines multiple causal experts:

- token n-gram expert
- within-word continuation expert
- byte-context next-token expert
- word-start expert
- suffix-of-current-word expert
- optional CTW token expert

At each scored position, the overlay aggregates evidence per candidate token, not just per expert. Agreement is weighted by observed reliability, gated by expected gain, and reduced when strong experts disagree.

### 3. Experimental in-model expert mixer

The current `train_gpt.py` now includes a small trainable `CausalExpertMixer` in the model itself. This is the smallest in-repo version of the “learned mixture-of-coders” idea.

It adds two lightweight learned experts:

- `RecentTokenReplayExpert`: biases logits toward tokens that followed recent exact repeats of the current token
- `NeuralCacheExpert`: biases logits toward tokens retrieved from recent hidden-state similarity in the prefix

These experts inject learned token-specific logit bias during both training and inference. The goal is to move some of the structure exploited by `online_best_agree` into the model proper instead of leaving it entirely as an eval-only overlay.

**Measurement status:** pending. The preserved March 31 result does not include this mixer.

### 4. Optional score-first TTT

Legal score-first TTT is still present in the file, but the default target configuration in this record keeps `TTT_ENABLED=0`. If enabled, TTT can run before or after the overlay via `TTT_ONLINE_AGREE_ORDER`.

---

## Architecture

| Component | Setting in this folder | Note / provenance |
|-----------|------------------------|-------------------|
| Layers | `11` layers, `512d`, `8` query heads, `4` KV heads | compact baseline family |
| MLP | `3x` expansion with `LeakyReLU(0.5)^2` | preserved from newer mainline stack |
| Attention | `XSA_LAST_N=11` | XSA on all 11 layers |
| BigramHash | `3072 x 112` | current mainline setting |
| RoPE | partial, `ROPE_DIMS=16` | compact-context tradeoff |
| LN scale | `1/sqrt(layer+1)` | enabled by default |
| VE | `VE_DIM=128`, `VE_LAYERS=7,8,9,10` | shared VE table with per-layer scales |
| SmearGate | enabled | position-mixing gate |
| U-Net skips | enabled | encoder-decoder skip connections |
| Attention sink | optional, default on | one learned sink token |
| Weight averaging | EMA + SWA | adaptive timing in current code |
| Optimizer | Parallel Muon + AdamW split | banked matrices + small replicated params |
| Quantization | Full Hessian GPTQ int6 | current code calibrates on training batches inside the counted training phase |
| Mixed precision export | `INT8_ATTN_LAYERS=0,-1` | preserve first/last attention layers |
| Compression | `lzma preset=6` | current code path |
| Overlay | online best-agree | prefix-only causal expert blend |
| Expert mixer | recent replay + neural cache | experimental, unmeasured |
| Flash Attention | FA3 / Hopper path | `flash_attn_interface` import |

---

## Requirements

**Flash Attention 3 on Hopper is required.** The script imports `flash_attn_interface` directly.

Typical dependencies:

```bash
pip install --break-system-packages sentencepiece
python3 -c "import sentencepiece; print('sentencepiece OK')"
python3 -c "from flash_attn_interface import flash_attn_func; print('fa3 OK')"
```

---

## RunPod 1x H100 Sanity Run

This runs the **current experimental code**, including the in-model expert mixer.

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-31_11L_BigramHash_XSAall_VE_GPTQ_CausalExpertOverlay_FA3

OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
RUN_ID=expert_mixer_2000_seed1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 XSA_LAST_N=11 VE_LAYERS=7,8,9,10 \
ATTENTION_SINK=1 \
EXPERT_MIXER_ENABLED=1 EXPERT_RECENT_WINDOW=32 EXPERT_CACHE_WINDOW=64 EXPERT_CACHE_DIM=64 \
USE_GPTQ=1 GPTQ_CALIB_SAMPLES=128 GPTQ_BLOCK_SIZE=64 INT8_ATTN_LAYERS=0,-1 \
TTT_ENABLED=0 ONLINE_BEST_AGREE_EVAL=1 EVAL_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=0 GPTQ_RESERVE_MS=10000 \
WARMDOWN_ITERS=1200 TIED_EMBED_LR=0.04 ITERATIONS=2000 TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

For the clean A/B against the same code path, rerun with:

```bash
EXPERT_MIXER_ENABLED=0
```

Each run keeps the existing `logs/<RUN_ID>.txt` file and now also mirrors master-process output into root-level seed logs:

- `train_seed<SEED>.log`
- `train.log` as an alias when `SEED=1337`

---

## RunPod 8x H100 Target Run

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-31_11L_BigramHash_XSAall_VE_GPTQ_CausalExpertOverlay_FA3

PYTHONUNBUFFERED=1 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 XSA_LAST_N=11 VE_LAYERS=7,8,9,10 \
ATTENTION_SINK=1 \
EXPERT_MIXER_ENABLED=1 EXPERT_RECENT_WINDOW=32 EXPERT_CACHE_WINDOW=64 EXPERT_CACHE_DIM=64 \
USE_GPTQ=1 GPTQ_CALIB_SAMPLES=128 GPTQ_BLOCK_SIZE=64 INT8_ATTN_LAYERS=0,-1 \
TTT_ENABLED=0 ONLINE_BEST_AGREE_EVAL=1 EVAL_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=600 GPTQ_RESERVE_MS=10000 \
WARMDOWN_ITERS=4000 TIED_EMBED_LR=0.04 ITERATIONS=6700 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Verification

Basic static verification for this folder:

```bash
python3 -m py_compile train_gpt.py
```

---

## Notes

- This record is best read as a serious **mainline experiment folder**, not as a completed PR-ready result.
- The preserved March 31 numbers are useful only as a baseline reference. They predate the current GPTQ-budget legality fix and do not cover the full sliding-window or overlay metrics.
- The newly added in-model expert mixer is the main new idea in the current code and should be evaluated first with a 1x H100 A/B before any 8x run.
- The current legality stance is narrower than “self-generated calibration”: GPTQ still uses training batches, but that work is now executed before evaluation and counted against the 600s training budget.
