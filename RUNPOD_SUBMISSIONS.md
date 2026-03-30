# RunPod Guide For New Submission Folders

This guide covers the three new submission folders added on `2026-03-29`:

- [2026-03-29_8192BPE_FactoredEmbed_ParallelMuon_TTT](/home/dev_rx/projects/parameter-golf/records/track_10min_16mb/2026-03-29_8192BPE_FactoredEmbed_ParallelMuon_TTT)
- [2026-03-29_ShadowQWarmdown_ParallelMuon](/home/dev_rx/projects/parameter-golf/records/track_10min_16mb/2026-03-29_ShadowQWarmdown_ParallelMuon)
- [2026-03-29_8192BPE_RecurrentFactored_ParallelMuon_TTT](/home/dev_rx/projects/parameter-golf/records/track_10min_16mb/2026-03-29_8192BPE_RecurrentFactored_ParallelMuon_TTT)

Use this on RunPod when you want to verify artifact size, final `val_bpb`, and whether a branch is worth a 3-seed record attempt.

## Before You Start

The repo README already documents the official Parameter Golf RunPod template and the leaderboard hardware requirement:

- final leaderboard runs must fit the `8x H100 SXM` budget in the challenge README
- the official Parameter Golf template already includes the Python dependencies

Reference:

- [README.md](/home/dev_rx/projects/parameter-golf/README.md#L124)
- https://docs.runpod.io/pods/configuration/use-ssh
- https://docs.runpod.io/pods/connect-to-a-pod

## 1. Create The Pod

In RunPod:

1. Add your SSH public key to your RunPod account.
2. Create a Pod from the official Parameter Golf template:
   `https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th`
3. For leaderboard-style verification, use `8x H100 SXM`.
4. Enable SSH access.
5. Connect using the SSH command shown in the Pod's `Connect` tab.

If you only want a cheap smoke test first, use `1x H100` and change `--nproc_per_node=8` to `--nproc_per_node=1`.

## 2. Clone Your Branch

From the Pod:

```bash
cd /workspace
git clone -b <your-branch> https://github.com/<your-user>/parameter-golf.git
cd parameter-golf
```

If these folders only exist locally right now, push them first from your machine before launching the pod.

## 3. Download The Datasets

Two of the new submissions use `sp8192`. One uses `sp1024`. Download both once:

```bash
cd /workspace/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
python3 data/cached_challenge_fineweb.py --variant sp8192
```

That creates:

- `./data/datasets/fineweb10B_sp1024`
- `./data/datasets/fineweb10B_sp8192`
- `./data/tokenizers/fineweb_1024_bpe.model`
- `./data/tokenizers/fineweb_8192_bpe.model`

## 4. General Run Pattern

Always `cd` into the specific record folder before running. The script writes `logs/*.txt` relative to the current working directory, which is what you want for submission packaging.

General pattern:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/<record-folder>
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

After each run, verify:

```bash
grep -E 'Total submission size int6\\+lzma|final_int6_roundtrip_exact|final_int6_sliding_window_exact|legal_ttt_exact' logs/<run_id>.txt
```

For the final claimed score:

- if `TTT_ENABLED=1`, use `legal_ttt_exact`
- if `TTT_ENABLED=0`, use `final_int6_sliding_window_exact`

## 5. Submission A: 8192 BPE + Factored Embedding

Folder:

- [train_gpt.py](/home/dev_rx/projects/parameter-golf/records/track_10min_16mb/2026-03-29_8192BPE_FactoredEmbed_ParallelMuon_TTT/train_gpt.py)

Run:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-29_8192BPE_FactoredEmbed_ParallelMuon_TTT

OMP_NUM_THREADS=1 \
RUN_ID=8192_factored_seed1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 EMBED_DIM=254 \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=3 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.02 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=500 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Quick verification:

```bash
grep -E 'embed_factorization|Total submission size int6\\+lzma|final_int6_sliding_window_exact|legal_ttt_exact' logs/8192_factored_seed1337.txt
```

Minimum 3-seed run set:

```bash
for seed in 1337 42 2025; do
  OMP_NUM_THREADS=1 \
  RUN_ID=8192_factored_seed${seed} \
  SEED=${seed} \
  DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model \
  VOCAB_SIZE=8192 EMBED_DIM=254 \
  NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
  MLP_MULT=3 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
  ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
  TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
  TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
  MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.02 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
  WARMDOWN_ITERS=3500 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
  ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=500 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

Collect results:

```bash
grep -H -E 'Total submission size int6\\+lzma|legal_ttt_exact' logs/8192_factored_seed*.txt
```

## 6. Submission B: ShadowQ Warmdown

Folder:

- [train_gpt.py](/home/dev_rx/projects/parameter-golf/records/track_10min_16mb/2026-03-29_ShadowQWarmdown_ParallelMuon/train_gpt.py)

Run:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-29_ShadowQWarmdown_ParallelMuon

OMP_NUM_THREADS=1 \
RUN_ID=shadowq_seed1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=3 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=500 \
SHADOW_Q_ENABLED=1 SHADOW_Q_START_SCALE=0.30 SHADOW_Q_SNAPSHOT_EVERY=50 \
SHADOW_Q_EXPORT=1 SHADOW_Q_AUTO_SELECT=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Quick verification:

```bash
grep -E 'shadow_q:|export_state:|Total submission size int6\\+lzma|final_int6_sliding_window_exact|legal_ttt_exact' logs/shadowq_seed1337.txt
```

Minimum 3-seed run set:

```bash
for seed in 1337 42 2025; do
  OMP_NUM_THREADS=1 \
  RUN_ID=shadowq_seed${seed} \
  SEED=${seed} \
  DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
  MLP_MULT=3 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
  ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
  TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
  TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
  MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
  WARMDOWN_ITERS=3500 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
  ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=500 \
  SHADOW_Q_ENABLED=1 SHADOW_Q_START_SCALE=0.30 SHADOW_Q_SNAPSHOT_EVERY=50 \
  SHADOW_Q_EXPORT=1 SHADOW_Q_AUTO_SELECT=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

Collect results:

```bash
grep -H -E 'shadow_q:selected|export_state:|Total submission size int6\\+lzma|legal_ttt_exact' logs/shadowq_seed*.txt
```

Notes:

- this is the lowest-priority branch for a leaderboard attempt
- watch `step_avg` carefully, because shadow snapshots can cost wallclock

## 7. Submission C: 8192 BPE + Recurrent Factored Moonshot

Folder:

- [train_gpt.py](/home/dev_rx/projects/parameter-golf/records/track_10min_16mb/2026-03-29_8192BPE_RecurrentFactored_ParallelMuon_TTT/train_gpt.py)

First run this one without TTT. The base sliding score matters more than the TTT score at this stage.

Base run:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-29_8192BPE_RecurrentFactored_ParallelMuon_TTT

OMP_NUM_THREADS=1 \
RUN_ID=8192_recurrent_base_seed1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model \
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
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=500 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Quick verification:

```bash
grep -E 'unique_blocks:|recurrent_loops:|ve_layers:|compile:model:|embed_factorization|Total submission size int6\\+lzma|final_int6_sliding_window_exact' logs/8192_recurrent_base_seed1337.txt
```

Minimum 3-seed base run set:

```bash
for seed in 1337 42 2025; do
  OMP_NUM_THREADS=1 \
  RUN_ID=8192_recurrent_base_seed${seed} \
  SEED=${seed} \
  DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model \
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
  ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=500 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

Collect base results:

```bash
grep -H -E 'Total submission size int6\\+lzma|final_int6_sliding_window_exact' logs/8192_recurrent_base_seed*.txt
```

If the base run is promising, rerun with TTT:

```bash
OMP_NUM_THREADS=1 \
RUN_ID=8192_recurrent_ttt_seed1337 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 EMBED_DIM=256 \
NUM_UNIQUE_BLOCKS=4 RECURRENT_LOOPS=3 LOOP_ADAPTER_RANK=8 \
MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=3 \
BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=2 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=auto \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
COMPILE_FULLGRAPH=0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.02 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=500 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Minimum 3-seed TTT run set:

```bash
for seed in 1337 42 2025; do
  OMP_NUM_THREADS=1 \
  RUN_ID=8192_recurrent_ttt_seed${seed} \
  SEED=${seed} \
  DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model \
  VOCAB_SIZE=8192 EMBED_DIM=256 \
  NUM_UNIQUE_BLOCKS=4 RECURRENT_LOOPS=3 LOOP_ADAPTER_RANK=8 \
  MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=3 \
  BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=2 ROPE_DIMS=16 LN_SCALE=1 \
  VE_ENABLED=1 VE_DIM=128 VE_LAYERS=auto \
  TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
  TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
  COMPILE_FULLGRAPH=0 \
  MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.02 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
  WARMDOWN_ITERS=3500 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
  ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=500 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

Collect TTT results:

```bash
grep -H -E 'Total submission size int6\\+lzma|legal_ttt_exact' logs/8192_recurrent_ttt_seed*.txt
```

## 8. Recommended Order

If you want to test all three efficiently on RunPod, use this order:

1. `8192BPE_RecurrentFactored_ParallelMuon_TTT` base run with `TTT_ENABLED=0`
2. `8192BPE_FactoredEmbed_ParallelMuon_TTT`
3. `8192BPE_RecurrentFactored_ParallelMuon_TTT` with `TTT_ENABLED=1` if the base score is competitive
4. `ShadowQWarmdown_ParallelMuon` only if you still want the side experiment

That order spends time first on the only branch with real moonshot upside.

Suggested seed set for every branch:

- `1337`
- `42`
- `2025`

## 9. After The Run

For a branch you want to submit:

1. Run at least 3 seeds.
2. Record the exact metric lines from each log.
3. Update that folder's `README.md`.
4. Update that folder's `submission.json`.
5. Keep the log files inside the record folder for the PR.

Relevant challenge rules:

- [README.md](/home/dev_rx/projects/parameter-golf/README.md#L182)
- [README.md](/home/dev_rx/projects/parameter-golf/README.md#L197)
