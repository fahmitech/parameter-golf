## 11L XSA4 + Partial RoPE + LN Scale + VE128 + LeakyReLU2 + EMA + GPTQ-lite

Fork-local candidate built from the proven March 22 record because the March 31 causal-expert line missed the 10-minute target. This folder is the current rerun candidate for `8xH100`, `600s`, and `<=16 MB`.

**Base reference from the source record:** `1.1233 val_bpb` (3-seed mean), `15.55 MB`, `~7100` steps in `600s`. Current fork metrics are pending a fresh rerun.

This folder should be read as a differentiated hypothesis on top of a validated starting point, not as a new measured submission. Its value is that it preserves a strong March 22 control line while introducing one small architectural change with positive local evidence from the March 23 family.

### Differentiated Hypothesis

This variant changes the MLP activation from `relu²` to `leaky_relu(0.5)²`.

Why this is the first real candidate worth testing:

- it is a one-line architectural change with essentially zero complexity overhead
- the same activation already has positive evidence in a neighboring 11-layer FA3 record
- it does not require changing the training regime, artifact format, or legality profile

Local evidence already present in this repo:

- the March 23 record reports `83.4ms/step`, so the broader family remains compatible with the 10-minute target
- its ablation table attributes about `-0.0021 BPB` to **LeakyReLU(0.5)²** in the same family
- the March 23 mean result is `1.1194`, showing that this activation participates in a line that already outruns the March 22 base

That does not prove this exact folder will beat the current SOTA. It does establish a credible, low-cost path: transplant a locally validated activation gain onto a stronger March 22 control and test whether the improvement survives in this cleaner stack.

### Key Innovations Over PR #374

Two novel post-training optimizations plus training hyperparameter tuning on top of PR #374's architecture:

| Change | PR #374 | This | Impact |
|--------|---------|------|--------|
| **GPTQ-lite** | Fixed clip (row max) | 5 clip percentiles per row, pick min MSE | -0.0006 BPB (zero training cost) |
| **EMA** | None (Tight SWA only) | EMA decay=0.997 every step | -0.0006 BPB (smoother averaging) |
| **Warmdown** | 3000 | 3500 | -0.0002 BPB |
| **Late QAT threshold** | 0.1 | 0.15 | -0.0001 BPB (earlier fake quant, smaller quant gap) |
| **Total** | **1.1246** | **1.1233** | **-0.0013 BPB** |

### GPTQ-lite: Per-Layer Optimal Clip Percentile Search

Instead of using the row maximum for int6 quantization scale, we try 5 clip percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0) per weight matrix row and pick the one minimizing reconstruction MSE. This is applied during post-training quantization with zero training cost.

### EMA Weight Averaging

Exponential moving average (decay=0.997) maintained every training step, applied before quantization. Stacks with Tight SWA — EMA provides continuous smoothing while SWA captures discrete checkpoints during warmdown.

### Base Reference Results (source record)

| Seed | Steps | val_loss | Sliding BPB (s64) | Artifact |
|------|-------|----------|-------------------|----------|
| **1337** | 7101 | 1.8958 | **1.1228** | 15.56 MB |
| 42 | ~7100 | 1.8972 | 1.1236 | 15.54 MB |
| 2024 | ~7100 | 1.8971 | 1.1236 | 15.59 MB |

**Mean: 1.1233 | Std: 0.0005** | Submitted: seed 1337 (best)

### Architecture (from PR #374)

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion (1536 hidden), `leaky_relu(0.5)^2` activation in this variant (`relu^2` in the March 22 control)
- U-Net skip connections (5 encoder, 6 decoder)
- Efficient Partial XSA on last 4 layers (GQA-aware, zero-alloc)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- Shared Value Embedding (dim=128, layers 9,10) with per-layer learned scales
- SmearGate + BigramHash (2048 buckets, dim=128)
- Tied embeddings, logit softcap=30.0

### Training

- FlashAttention 3 (Hopper-optimized)
- Muon optimizer (matrices): lr=0.025, momentum=0.99 (warmup 0.92->0.99 over 1500 steps), WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3500 iterations (wallclock-based)
- **EMA**: decay=0.997, every step
- **Tight SWA**: every 50 steps when scale<0.2
- **Late QAT**: STE int6 fake-quantization when LR scale<0.15
- OrthoInit + muP-scaled output projections

### Quantization

- **GPTQ-lite**: Per-row optimal clip percentile search (5 candidates) for int6
- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

### Validation Thesis

This folder is a low-entropy architectural bet. The near-term goal is to show that we can carry one already-supported improvement into a stronger control line and obtain a candidate that plausibly contends with the current frontier without sacrificing runtime discipline.

Why this base matters:

- **It has a hard external performance anchor.** The source March 22 record already demonstrates `1.1233 val_bpb`, `15.55 MB`, and about `7100` steps in `600s` on `8xH100`.
- **It lives inside a strong family.** Closely related 11-layer FA3 records in the same lineage improve further, including a March 23 variant at `1.1194` with legal TTT.
- **The fork is intentionally low-entropy.** Almost all architectural behavior is unchanged from the March 22 source line; the main delta is the MLP activation, plus operational cleanup for reruns and logging.
- **It keeps the control intact.** Because the change surface is so small, any measured movement is easier to interpret than it would be in a broad multi-change branch.

What is actually new here:

- `leaky_relu(0.5)²` as the default MLP activation for this variant
- stable seed-level logging for reruns
- clearer hardware/runbook instructions
- explicit positioning of this folder as a low-cost differentiated candidate rather than a broad rewrite

What is not new here:

- no measured architecture improvement over the March 22 source record yet
- no new submission metric from this fork yet
- no proof yet that the activation gain transfers one-for-one into this stack

The investable story is therefore disciplined execution:

- **Stage 1:** validate that the March 22 control still behaves correctly in this fork
- **Stage 2:** measure whether `LeakyReLU²` preserves runtime while improving score
- **Stage 3:** stop quickly if runtime, legality, artifact size, or quality miss predeclared bounds

Suggested paid-run decision gates:

- `GO`: the reproduced run stays within the 10-minute budget, artifact remains `<16 MB`, and the final score lands within about `0.005-0.010 BPB` of the source record
- `NO-GO`: step time is materially off target on `8xH100`, artifact size breaks the cap, legality becomes questionable, or reproduced quality misses badly enough that additional runs have poor expected value

If this line earns further capital, the next differentiated ideas should be treated as follow-on hypotheses on top of this activation variant, not mixed into the same test.

### Run Command

Before launching `torchrun`, verify how many GPUs the container can actually see:

```bash
nvidia-smi -L
python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
echo "$CUDA_VISIBLE_DEVICES"
```

Set `--nproc_per_node` to the visible GPU count. If you request `8` but the box only exposes `1` or `4`, PyTorch will fail with `CUDA error: invalid device ordinal`.

For the target `8xH100` run:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-02_11L_XSA4_PartialRoPE_LNScale_VE128_LeakyReLU2_EMA_GPTQlite

OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
RUN_ID=base_11l_gptqlite_seed1337 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_ACTIVATION=leaky_relu2 \
MLP_LEAKY_SLOPE=0.5 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For an A/B against the original March 22 activation inside the same code path:

```bash
MLP_ACTIVATION=relu2
```

If the container only sees a single GPU, do not treat that box as a submission reproduction environment. A longer 1-GPU run is still only a sanity check for startup, loss descent, export, and logging.

For a 2000-iteration 1-GPU debug run, use:

```bash
OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
RUN_ID=debug_1gpu_2000_seed1337 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_ACTIVATION=leaky_relu2 \
MLP_LEAKY_SLOPE=0.5 \
ITERATIONS=2000 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Use the 1-GPU debug run only to answer:

- does the script launch cleanly
- does `world_size:1` appear as expected
- does training loss trend downward
- does a longer run stay numerically stable
- does the activation-specific log line confirm `mlp_activation:leaky_relu2`
- do `logs/<RUN_ID>.txt`, `train_seed<SEED>.log`, and `train.log` get written

Do not use the resulting step time or BPB as a decision-quality estimate for the intended `8xH100, 600s` submission regime.

### Cheap 1-GPU A/B Protocol

Use the same cheap regime for both the March 22 control and this LeakyReLU2 variant. Change only `MLP_ACTIVATION`.

Control (`relu^2`):

```bash
OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
RUN_ID=ab_relu2_seed1337 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_ACTIVATION=relu2 \
ITERATIONS=2000 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Variant (`leaky_relu(0.5)^2`):

```bash
OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
RUN_ID=ab_leakyrelu2_seed1337 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_ACTIVATION=leaky_relu2 \
MLP_LEAKY_SLOPE=0.5 \
ITERATIONS=2000 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Compare only these four numbers from the two logs:

- `step_avg` at step `2000`
- `step:2000 ... val_bpb`
- `DIAGNOSTIC post_ema ... val_bpb`
- `final_int6_roundtrip_exact ... val_bpb`

Use the A/B only as a local ranking signal between activations. It is still not a submission-quality estimate.

Provisional decision rule:

- `KEEP` the LeakyReLU2 variant if it is no more than about `3%` slower and wins by at least `0.005 BPB` on either `post_ema` or `final_int6_roundtrip_exact` without a material loss on the other.
- `KILL` the LeakyReLU2 variant if it is slower and does not clearly improve any of the score checkpoints, or if it worsens both `post_ema` and `final_int6_roundtrip_exact`.
- `INCONCLUSIVE` if the score movement is within about `0.003 BPB`; in that case, prefer the simpler March 22 control for paid 8-GPU runs.

Defaults in `train_gpt.py` already encode the March 22 stack plus this variant's activation choice: `NUM_LAYERS=11`, `XSA_LAST_N=4`, `ROPE_DIMS=16`, `LN_SCALE=1`, `VE_ENABLED=1`, `VE_LAYERS=9,10`, `WARMDOWN_ITERS=3500`, `LATE_QAT_THRESHOLD=0.15`, `EVAL_STRIDE=64`, and `MLP_ACTIVATION=leaky_relu2`.

Each run writes:

- `logs/<RUN_ID>.txt`
- `train_seed<SEED>.log`
- `train.log` when `SEED=1337`

### Reproducibility

All 3 seeds produce valid artifacts under 16MB with tight variance (std=0.0005 BPB). The GPTQ-lite clip search is deterministic.
