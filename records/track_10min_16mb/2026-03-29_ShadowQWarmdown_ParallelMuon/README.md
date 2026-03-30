# ShadowQ Warmdown + Parallel Muon

Unbenchmarked submission scaffold derived from the March 23, 2026 `LeakyReLU_LegalTTT_ParallelMuon` record stack.

This variant does not copy the later self-generated GPTQ submission. Its main change is an export-aware warmdown path:

- During late warmdown, the banked attention/MLP tensors that are actually exported to int6 are fake-quantized in training.
- Every `SHADOW_Q_SNAPSHOT_EVERY` steps in that phase, the current model is quantized, dequantized, and accumulated into a shadow average.
- At the end, the script evaluates both the standard EMA path and the shadow-quantized average, and can export the better of the two when `SHADOW_Q_AUTO_SELECT=1`.

The intent is to reduce the train/export mismatch on the exact banked tensors that dominate the final int6 artifact, without copying the competitor GPTQ recipe.

## Status

- `val_bpb`: pending
- artifact bytes: pending
- verification: `python3 -m py_compile train_gpt.py`

## Key Delta

The March 23 stack already had late QAT for small `CastedLinear` modules. This submission extends the idea to the banked tensors that drive the real export path:

- `qo_bank`
- `kv_bank`
- `mlp_up_bank`
- `mlp_down_bank`

Those banks are fake-quantized only during late warmdown, then averaged through a quantized roundtrip shadow path before final export.

## Suggested Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SHADOW_Q_ENABLED=1 SHADOW_Q_START_SCALE=0.30 SHADOW_Q_SNAPSHOT_EVERY=50 \
SHADOW_Q_EXPORT=1 SHADOW_Q_AUTO_SELECT=1 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Notes

- The final log still emits the standard `final_int6_roundtrip_exact`, `final_int6_sliding_window_exact`, and `legal_ttt_exact` lines for submission parsing.
- This folder is a candidate record submission, not a claimed result. Fill in measured metrics only after 3-seed verification.
