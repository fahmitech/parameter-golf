# 8192 BPE + Factored Tied Embedding + Parallel Muon + Legal TTT

Submission scaffold derived from [2026-03-23_LeakyReLU_LegalTTT_ParallelMuon](/home/dev_rx/projects/parameter-golf/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon), with the first planned ablation implemented:

- switch from the `sp1024` dataset/tokenizer to `sp8192`
- replace the full `vocab_size x model_dim` tied embedding with a factored tied embedding
- keep the March 23 banked transformer trunk, Parallel Muon optimizer path, mixed int6 export, sliding eval, and legal score-first TTT

## Status

This folder is a runnable submission scaffold, not a benchmarked record yet.

- `val_bpb`: pending measurement
- artifact size: pending measurement
- statistical significance: not established

## Change Summary

The main architectural change is embedding factorization:

- token table: `VOCAB_SIZE x EMBED_DIM`
- forward path: `tok_emb -> embed_proj -> model trunk`
- tied logits path: `hidden -> embed_proj_rev -> tok_emb.weight`

Default settings for this experiment:

- `DATA_PATH=./data/datasets/fineweb10B_sp8192`
- `TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model`
- `VOCAB_SIZE=8192`
- `EMBED_DIM=254`
- March 23 stack otherwise preserved by default

The intent is to buy a much larger tokenizer while keeping the artifact inside 16MB and preserving the proven training/eval systems from the March 23 run.

## Run Command

```bash
RUN_ID=8192_factored_embed_a1 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 EMBED_DIM=254 \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=3 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.02 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Outcome

If this direction works, it should improve the compression side of the score by giving the model access to a much larger token vocabulary without paying the full `8192 x 512` tied-embedding cost.

If it fails, the likely causes are:

- optimizer retuning needed for the lower-dimensional embedding space
- `EMBED_DIM=254` is too aggressive for the March 23 trunk
- the March 23 auxiliary stack was tuned too tightly around the `sp1024` regime

## Next Ablations

After the first run, the next useful sweeps are:

1. `EMBED_DIM`: `192, 224, 254, 320`
2. `TIED_EMBED_LR`: `0.012, 0.02, 0.03`
3. `BIGRAM_VOCAB_SIZE`: `0, 1536, 3072`
4. `EVAL_STRIDE`: `64, 32, 16`
5. legal TTT on/off after the pre-TTT quality is measured
