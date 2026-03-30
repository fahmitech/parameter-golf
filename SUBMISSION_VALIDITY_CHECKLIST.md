# Submission Validity Checklist

Use this before trusting any claimed `val_bpb`, especially for novel eval-time methods.

The goal is simple: reject invalid scores early, before spending H100 time reproducing them.

## 1. Context-Only Prediction

For every scored position, the predicted distribution must depend only on past context.

It must not depend on:

- the current target token
- future validation tokens
- any hidden label-derived signal

Hard rule:

- changing `target_ids` while keeping `input_ids` fixed must not change the predicted logits or probabilities

If a method uses `target_ids` anywhere before the final `cross_entropy` or `gather`, treat it as suspicious immediately.

Typical invalid pattern:

```python
logits = f(input_ids, target_ids)
```

That is only valid if `target_ids` is used strictly after logits are fixed, for scoring only.

## 2. Probability Validity

At each scored position, the model must define a valid probability distribution over the whole vocabulary.

Checks:

1. compute the full distribution over all vocab tokens
2. verify all probabilities are finite
3. verify all probabilities are non-negative
4. verify `abs(probs.sum() - 1.0) < 1e-4`

If a method only computes a corrected score for the gold token and never defines the rest of the distribution, the BPB is not trustworthy.

## 3. No Future-Token Leakage

Eval-time adaptation is only valid if it uses tokens already scored.

Invalid patterns:

- pass 2 rescoring of token `t` using caches built from tokens `> t`
- building n-gram or neural caches from the full validation document before scoring token `t`
- any bidirectional or full-document statistic used during scoring of earlier positions

Sanity question:

- when scoring token `t`, could this code have been run online left-to-right without seeing `t+1 ... T`?

If not, it is invalid.

## 4. Target-Conditioned Masking Is Invalid

This deserves its own check because it is easy to miss.

Invalid example:

```python
implausible.scatter_(-1, target_ids.unsqueeze(-1), False)
```

Why invalid:

- it changes the predicted distribution using the correct answer
- it leaks label information directly into the logits

Any masking, gating, reranking, or calibration step that depends on the gold token before scoring is invalid.

## 5. Same Inputs, Same Logits

Predicted logits should be deterministic given:

- model parameters
- past tokens
- allowed eval-time state derived causally from past scored tokens

Quick test:

1. fix `input_ids`
2. run the prediction twice with different `target_ids`
3. compare logits

Expected:

- logits must be identical

If they differ, the evaluation path is invalid.

## 6. Full-Vocab Check For Mixing Methods

For any method that modifies logits or probabilities at eval time:

- TTT
- TARA-style contrastive methods
- n-gram caches
- retrieval or memory mixing
- heuristic rerankers

do both checks:

1. target-independence
2. full-vocab normalization

Do not trust “looks causal” by inspection.

## 7. Eval-Time State Budget

Even if a method is mathematically valid, check whether it is still in the spirit of the challenge.

Track:

- extra model weights created during eval
- caches not derivable from the artifact alone
- large hidden-state stores
- retrieval tables, hash tables, or side structures

Questions:

- how many extra bytes are created during eval?
- is the method still under the 10-minute evaluation budget?
- is the extra state causal and defensible?

This does not always make a submission invalid, but it may make it non-competitive or contestable.

## 8. Consistency Checks

Compare the prose, config table, and code defaults.

Red flags:

- README says `MLP 2.5x`, code uses `MLP_MULT=3.0`
- README says `stride 128`, code uses `EVAL_STRIDE=64`
- claimed runtime does not match training + eval times

These do not prove invalidity, but they are strong signals to inspect the actual implementation more closely.

## 9. Minimal Practical Test Suite

Before believing any new submission, run these checks:

1. Target-independence test
   Keep `input_ids` fixed, vary `target_ids`, assert logits unchanged.
2. Full-distribution sum test
   Assert per-position probabilities sum to `1`.
3. Causality review
   Inspect whether any future tokens are used for current-token scoring.
4. Timing check
   Verify train time and eval time from logs against the claimed budget.
5. Artifact check
   Verify total bytes from the actual output files and logged totals.

## 10. Decision Rule

Treat a submission as invalid unless all of these are true:

- predictions are target-independent
- probabilities are valid and normalized
- no future-token leakage occurs
- runtime and artifact claims match the actual logs

If any one of those fails, the claimed `val_bpb` is not comparable to the leaderboard.
