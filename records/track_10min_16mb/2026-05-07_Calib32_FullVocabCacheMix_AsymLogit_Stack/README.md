# Calib32 Full-Vocab Cache Mix + AsymLogit Stack

**Status:** implementation-ready experiment. The code is runnable as a record folder, but the final `8xH100` score and three-seed logs still need to be produced before opening a leaderboard PR.

This folder forks the FA3/XSA-all/GPTQ/online-overlay line and adds a stricter causal evaluation mode. The May 7 update turns the single cache into a causal compression arbiter: the artifact stores the rule for growing prefix-only experts during evaluation, then mixes those experts into the base model with a capped normalized distribution.

```text
p_final(. | prefix) = (1 - lambda(prefix)) * p_model(. | prefix)
                    + lambda(prefix)       * q_ngram(. | prefix)
```

`q_ngram` is a real full-vocabulary smoothed token n-gram distribution, not a gold-token-only correction. For each order `d`:

```text
q_d(a | ctx) = (count(ctx, a) + alpha) / (count(ctx) + alpha * vocab_size)
```

The arbiter can also add `q_recent`, a recent-prefix unigram distribution:

```text
q_recent(a) = (recent_count(a) + alpha) / (window_tokens + alpha * vocab_size)
```

The evaluator mixes active expert distributions with weights computed only from prefix statistics. Scoring gathers `q(target)` after each distribution and mixture weight are defined. This is designed to satisfy the prequential validity checks:

- strict prefix-only dependence
- full normalized distribution over the vocabulary
- score-before-update semantics
- single left-to-right pass

This refinement also adds `online_full_vocab_cache.c`, a native implementation of the same full-vocabulary arbiter state. It keeps the Python `process_chunk()` contract (`q(target)`, `lambda`, top-prob diagnostics, hit/order metadata), builds on demand with `gcc`, and falls back to the Python dictionaries if the native path is disabled or unavailable.

## Main Delta

The new code paths are enabled by default:

```bash
CAUSAL_ARBITER_EVAL=1
ONLINE_FULL_VOCAB_MIX_EVAL=1
PREFIX_MIX_TTT_EVAL=1
TTT_ENABLED=1
ASYM_LOGIT_RESCALE=1
NATIVE_FULL_VOCAB_CACHE=1
```

It logs:

```text
legal_prefix_mix_ttt_sliding_window_exact val_loss:... val_bpb:...
legal_prefix_mix_ttt_compare ...
legal_prefix_mix_ttt_timing ...
```

The previous `ONLINE_BEST_AGREE_EVAL` path remains available but defaults off in this folder.

The causal arbiter currently has two prefix-grown experts:

- `ngram`: the existing full-vocab smoothed context cache over orders `4,8,16,32`
- `recency`: a recent-prefix unigram memory for bursty names, terms, punctuation, and repeated local tokens

The arbiter caps total expert probability mass, renormalizes the expert mixture, and optionally damps cache trust when the base model's own top probability is high. This keeps the final distribution normalized:

```text
p_final = (1 - lambda_total) * p_model
        + lambda_total       * q_arbiter
```

`ASYM_LOGIT_RESCALE=1` adds trainable positive and negative softcap scalars to the model. The same asymmetric softcap is used in training and eval, so this avoids the common calibration mismatch where only the eval path gets asymmetric clipping.

## Key Environment Knobs

```bash
ONLINE_FULL_VOCAB_MIX_EVAL=1
PREFIX_MIX_TTT_EVAL=1
TTT_ENABLED=1
ASYM_LOGIT_RESCALE=1
CAUSAL_ARBITER_EVAL=1
NATIVE_FULL_VOCAB_CACHE=1
NATIVE_FULL_VOCAB_TABLE_BITS=26
ARBITER_NGRAM_ENABLED=1
ARBITER_RECENCY_ENABLED=1
ARBITER_MAX_LAMBDA=0.35
ARBITER_MODEL_CONFIDENCE_POWER=0.5
ARBITER_MODEL_CONFIDENCE_FLOOR=0.10
USE_GPTQ=1
GPTQ_CALIB_SAMPLES=32
FULL_VOCAB_CACHE_ORDERS=4,8,16,32
FULL_VOCAB_CACHE_ALPHA=0.05
FULL_VOCAB_CACHE_THRESHOLD=0.28
FULL_VOCAB_CACHE_MAX_LAMBDA=0.35
FULL_VOCAB_CACHE_WEIGHT_POWER=1.5
RECENCY_CACHE_WINDOW=4096
RECENCY_CACHE_ALPHA=0.05
RECENCY_CACHE_THRESHOLD=0.08
RECENCY_CACHE_MAX_LAMBDA=0.12
RECENCY_CACHE_WEIGHT_POWER=1.25
FULL_VOCAB_CACHE_CHUNK_TOKENS=131072
TTT_CHUNK_TOKENS=32768
TTT_EPOCHS=3
TTT_LR=0.002
BATCH_SEQS=32
```

Conservative ablations:

- Lower `FULL_VOCAB_CACHE_MAX_LAMBDA` first if the mixture over-trusts cache repeats.
- Set `ARBITER_RECENCY_ENABLED=0` to recover the previous n-gram-only full-vocab cache path.
- Set `ARBITER_MODEL_CONFIDENCE_POWER=0` to disable base-model confidence damping.
- Raise `FULL_VOCAB_CACHE_THRESHOLD` if rare contexts cause regressions.
- Try `FULL_VOCAB_CACHE_ORDERS=8,16,32` if short-order cache hurts punctuation or boilerplate.

## Run Command

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-05-07_Calib32_FullVocabCacheMix_AsymLogit_Stack

OMP_NUM_THREADS=1 \
RUN_ID=full_vocab_cache_seed1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
USE_GPTQ=1 \
GPTQ_CALIB_SAMPLES=32 \
ASYM_LOGIT_RESCALE=1 \
CAUSAL_ARBITER_EVAL=1 \
ONLINE_FULL_VOCAB_MIX_EVAL=1 \
PREFIX_MIX_TTT_EVAL=1 \
TTT_ENABLED=1 \
ONLINE_BEST_AGREE_EVAL=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Use the `legal_prefix_mix_ttt_sliding_window_exact` line as the experimental score for this folder.

## Legality Notes

For target position `t`, the cache distribution is computed from counts accumulated before token `t` is inserted. The mixture weight is a function of cache confidence and does not depend on the realized target. After the score for `t` is fixed, the token is inserted into the cache and can only influence future positions.

When `TTT_ENABLED=1`, the target path is `PREFIX_MIX_TTT_EVAL=1`: for each chunk, the script scores with the current model plus prefix cache mixture, then updates model parameters on that already-scored chunk, then moves to the next chunk. It does not run the cache score as a second pass after a full validation-set TTT adaptation.

The implementation stores only prefix-derived count tables during evaluation. It does not access training data during evaluation, make network calls, or precompute statistics from future validation tokens for earlier scores.

## Verification Checklist

Before PR:

1. Run one smoke pass and confirm `ast ok` / import / startup on the RunPod image.
2. Compare `legal_prefix_mix_ttt_compare gain_bpb` against the base sliding-window line.
3. Confirm `legal_prefix_mix_ttt_timing wallclock` keeps total eval under 10 minutes.
4. Run at least three seeds and fill in `submission.json`.
5. Attach logs with `Total submission size int6+lzma` and the exact online score line.
