# Parameter Golf: Comprehensive Implementation Plan
## Target: val_bpb ≤ 1.0 | Budget: 16MB artifact | Time: 10min on 8×H100

---

## Executive Summary

The current SOTA (1.1194 bpb) is a well-engineered standard transformer with many
bolt-on improvements. To reach 1.0 bpb, we need **architectural paradigm shifts**,
not incremental tuning. This plan implements five synergistic changes in priority order:

| Phase | Change | Expected Δ bpb | Cumulative |
|-------|--------|---------------|------------|
| 0 | Baseline reproduction | — | 1.1194 |
| 1 | Depth Recurrence (4 blocks × 8 loops) | −0.035 | ~1.084 |
| 2 | 8K Vocabulary + byte-fallback BPE | −0.025 | ~1.059 |
| 3 | Mixture-of-Experts MLP (2 experts, top-1) | −0.015 | ~1.044 |
| 4 | Fisher-weighted mixed int4/int6/int8 quant | −0.010 | ~1.034 |
| 5 | Advanced TTT + eval improvements | −0.020 | ~1.014 |
| 6 | Integration tuning + hyperparameter sweep | −0.010 | ~1.004 |

---

## Phase 0: Baseline Reproduction & Instrumentation (Day 1)

### 0.1 Environment Setup

```bash
# On 8×H100 RunPod instance
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

# Verify baseline reproduction
RUN_ID=baseline_verify \
torchrun --standalone --nproc_per_node=8 train_gpt.py
# Expected: val_bpb ≈ 1.22 (naive baseline)
```

### 0.2 Reproduce Current SOTA

Copy the PR #549 `train_gpt.py` and verify 1.1194 bpb across 3 seeds.
Record per-step training loss curves, peak memory, step timings.

### 0.3 Add Instrumentation

Add these diagnostic hooks (remove before final submission):

```python
# Add to main() after each validation
def log_diagnostics(model, step):
    """Log per-layer gradient norms, activation norms, weight utilization."""
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.float().norm().item()
            weight_norm = p.data.float().norm().item()
            log0(f"diag step:{step} {name} grad_norm:{grad_norm:.6f} weight_norm:{weight_norm:.4f}")
```

**Deliverable**: Baseline log with diagnostics, 3-seed mean ± std for val_bpb.

---

## Phase 1: Depth Recurrence (Days 2-4)

### 1.1 Architectural Design

Replace the current 11 unique transformer blocks with **4 unique blocks looped 8× each = 32 effective layers**.

```
Current:  Block_0 → Block_1 → ... → Block_10  (11 unique)
Proposed: [Block_0 → Block_1 → Block_2 → Block_3] × 8  (4 unique, 32 effective)
```

Each loop iteration `t ∈ [0, 7]` injects a lightweight adapter:

```python
class LoopAdapter(nn.Module):
    """Per-iteration conditioning for depth recurrence."""
    def __init__(self, dim: int, num_loops: int, rank: int = 8):
        super().__init__()
        # Learned iteration embedding (added to residual)
        self.iter_embed = nn.Embedding(num_loops, dim)
        nn.init.zeros_(self.iter_embed.weight)

        # Low-rank residual: down-project → nonlinearity → up-project
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.orthogonal_(self.down.weight)
        nn.init.zeros_(self.up.weight)

        # Scalar gate (initialized near zero so early training = skip)
        self.gate = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(self, x: Tensor, loop_idx: int) -> Tensor:
        adapt = self.up(F.silu(self.down(x)))
        return x + self.gate * (self.iter_embed.weight[loop_idx] + adapt)
```

### 1.2 Model Dimension Increase

Freed parameters from weight sharing allow wider model:

```
Current:  11 layers × dim=512 → 4 bank types × 11 slices each
Proposed: 4 layers × dim=768  → 4 bank types × 4 slices each

Parameter budget comparison:
  Current banks:  11 × (512×512 + 512×512 + 256×512 + 256×512 + 1536×512 + 512×1536)
                = 11 × 2,359,296 = ~25.9M params in banks

  Proposed banks: 4 × (768×768 + 768×768 + 256×768 + 256×768 + 2304×768 + 768×2304)
                = 4 × 5,308,416 = ~21.2M params in banks
                + 32 adapters × ~12K params each = ~384K
                = ~21.6M total (saves ~4.3M params → available for vocab/embedding)
```

### 1.3 U-Net Skip Connections for Recurrent Depth

Adapt the existing U-Net skip architecture to work within each loop:

```python
class RecurrentGPT(nn.Module):
    def __init__(self, ...):
        # 4 unique blocks
        self.num_unique_blocks = 4
        self.num_loops = 8
        self.blocks = nn.ModuleList([Block(...) for _ in range(self.num_unique_blocks)])

        # 4 unique blocks → 2 encoder + 2 decoder per loop
        self.num_encoder_per_loop = 2
        self.num_decoder_per_loop = 2

        # Per-loop adapters (one per effective layer position)
        self.adapters = nn.ModuleList([
            LoopAdapter(dim, self.num_loops, rank=8)
            for _ in range(self.num_unique_blocks)
        ])

        # Skip weights: one per decoder block per loop
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_loops, self.num_decoder_per_loop, dim)
        )

        # Loop-level mixing gate: controls how much each loop contributes
        self.loop_gates = nn.Parameter(torch.ones(self.num_loops))

    def forward_trunk(self, x: Tensor, x0: Tensor, input_ids: Tensor):
        v0 = None
        for loop_idx in range(self.num_loops):
            # --- Encoder half ---
            skips = []
            for bi in range(self.num_encoder_per_loop):
                x = self.adapters[bi](x, loop_idx)
                # index into banks: use unique block index
                x, raw_v = self.blocks[bi](x, x0, ...)
                if v0 is None and raw_v is not None:
                    v0 = raw_v
                skips.append(x)

            # --- Decoder half ---
            for bi in range(self.num_decoder_per_loop):
                block_idx = self.num_encoder_per_loop + bi
                if skips:
                    skip_w = self.skip_weights[loop_idx, bi]
                    x = x + skip_w[None, None, :] * skips.pop()
                x = self.adapters[block_idx](x, loop_idx)
                x, _ = self.blocks[block_idx](x, x0, ..., v0=v0)

        return x
```

### 1.4 Bank Restructuring

The parameter banks must be restructured for 4 unique blocks instead of 11:

```python
# In __init__:
n = self.num_unique_blocks  # 4 instead of 11
head_dim = model_dim // num_heads
kv_dim = num_kv_heads * head_dim
mlp_dim = int(mlp_mult * model_dim)

self.qo_bank = nn.Parameter(torch.empty(2 * n, model_dim, model_dim))  # Q and Out
self.kv_bank = nn.Parameter(torch.empty(2 * n, kv_dim, model_dim))     # K and V
self.mlp_up_bank = nn.Parameter(torch.empty(n, mlp_dim, model_dim))
self.mlp_down_bank = nn.Parameter(torch.empty(n, model_dim, mlp_dim))
```

### 1.5 Training Considerations

**Gradient flow**: With 32 effective layers, gradient magnitude at early layers drops.
Mitigations:
- The per-loop adapters have skip connections (gate initialized near zero)
- The U-Net skips provide shortcut gradient paths
- Use gradient clipping per-loop-iteration (not just global)

**Compilation**: `torch.compile` with loops requires `dynamic=True` or loop unrolling.
Test both approaches:

```python
# Option A: Unroll loops for torch.compile (preferred if memory allows)
# In forward, manually unroll 8 iterations

# Option B: Use torch.compile only on individual block forward
compiled_block = [torch.compile(b, dynamic=False) for b in self.blocks]
```

### 1.6 Validation

- [ ] Verify recurrent model matches parameter budget (< 22M in banks)
- [ ] Verify forward pass produces correct output shape
- [ ] Train for 200 steps, confirm loss decreases
- [ ] Full 10-min training run, compare val_bpb to SOTA
- [ ] Profile step time — must not exceed 2× current step time

**Risk**: If depth recurrence is too slow per step, reduce to 6 loops (24 effective layers).

---

## Phase 2: 8K Vocabulary (Days 5-6)

### 2.1 Tokenizer Training

Train a new 8192-token BPE tokenizer on the FineWeb training data:

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='fineweb_train_text.txt',  # Extract text from shards
    model_prefix='fineweb_8192_bpe',
    vocab_size=8192,
    model_type='bpe',
    character_coverage=1.0,
    byte_fallback=True,  # Critical: handles all Unicode
    normalization_rule_name='identity',  # No normalization
    max_sentence_length=16384,
    num_threads=64,
    train_extremely_large_corpus=True,
)
```

### 2.2 Data Re-tokenization

```bash
# Re-tokenize the full FineWeb dataset with 8192 vocab
python3 data/retokenize.py \
    --input-dir ./data/datasets/fineweb10B_sp1024/ \
    --output-dir ./data/datasets/fineweb10B_sp8192/ \
    --tokenizer ./data/tokenizers/fineweb_8192_bpe.model \
    --vocab-size 8192
```

Script outline for `retokenize.py`:

```python
def retokenize_shard(input_path, output_path, sp):
    """Read uint16 tokens from old shard, decode to text, re-encode with new tokenizer."""
    old_tokens = load_data_shard(input_path)
    # Decode using old tokenizer
    old_sp = spm.SentencePieceProcessor(model_file=OLD_TOKENIZER)
    text = old_sp.decode(old_tokens.tolist())
    # Re-encode with new tokenizer
    new_tokens = sp.encode(text)
    # Write new shard with same header format
    write_data_shard(output_path, new_tokens)
```

### 2.3 Embedding Table Budget

With 8192 vocab and dim=768:

```
Embedding: 8192 × 768 = 6,291,456 params
At int4 quantization: ~3.1 MB
At int6 quantization: ~4.7 MB
At int8 quantization: ~6.3 MB
```

With tied embeddings, this is shared with the output head.
At int6, this fits within budget while leaving ~8-9MB for transformer weights.

### 2.4 BPB Calculation Verification

**Critical**: The val_bpb metric counts bytes, not tokens. With a larger vocabulary:
- Fewer tokens per byte → lower tokens_per_byte ratio
- Each token prediction has higher entropy (more choices)
- Net effect: significant bpb improvement if the model maintains similar per-token accuracy

Must verify `build_sentencepiece_luts()` correctly handles the new tokenizer.

```python
# Verification script
sp_new = spm.SentencePieceProcessor(model_file='fineweb_8192_bpe.model')
sp_old = spm.SentencePieceProcessor(model_file='fineweb_1024_bpe.model')

test_text = "The quick brown fox jumps over the lazy dog."
old_tokens = sp_old.encode(test_text)
new_tokens = sp_new.encode(test_text)
text_bytes = len(test_text.encode('utf-8'))

print(f"Old: {len(old_tokens)} tokens, {len(old_tokens)/text_bytes:.3f} tokens/byte")
print(f"New: {len(new_tokens)} tokens, {len(new_tokens)/text_bytes:.3f} tokens/byte")
# Expected: new tokenizer produces ~30-40% fewer tokens
```

### 2.5 Hyperparameter Adjustments for 8K Vocab

```python
# Larger vocab → more embedding params → adjust learning rates
VOCAB_SIZE = 8192
TIED_EMBED_LR = 0.02      # Lower than 0.035 (more params to train)
TIED_EMBED_INIT_STD = 0.003  # Tighter init for larger table
LOGIT_SOFTCAP = 50.0      # Increase for more vocab classes
```

### 2.6 Validation

- [ ] New tokenizer produces valid re-tokenized shards
- [ ] `build_sentencepiece_luts()` works correctly with 8192 vocab
- [ ] val_bpb improves over 1024 vocab with same model architecture
- [ ] Embedding table fits within 16MB budget after quantization

---

## Phase 3: Mixture-of-Experts MLP (Days 7-8)

### 3.1 Architecture

Replace the single MLP with a 2-expert MoE with top-1 routing:

```python
class MoEMLP(nn.Module):
    """2-expert Mixture of Experts MLP with top-1 routing."""
    def __init__(self, dim: int, mlp_mult: float, num_experts: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.expert_dim = int(mlp_mult * dim) // num_experts  # Each expert is half-width
        # Router: simple linear projection to num_experts
        self.router = nn.Linear(dim, num_experts, bias=False)
        nn.init.zeros_(self.router.weight)
        # Temperature for router softmax
        self.router_temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: Tensor, up_ws: list[Tensor], down_ws: list[Tensor]) -> Tensor:
        """
        x: [B, T, D]
        up_ws: list of num_experts tensors, each [expert_dim, D]
        down_ws: list of num_experts tensors, each [D, expert_dim]
        """
        B, T, D = x.shape
        # Compute routing weights
        router_logits = self.router(x.detach()) / self.router_temp.abs().clamp(min=0.1)
        # Top-1 routing
        expert_idx = router_logits.argmax(dim=-1)  # [B, T]
        routing_weights = F.softmax(router_logits, dim=-1)
        top1_weight = routing_weights.gather(-1, expert_idx.unsqueeze(-1)).squeeze(-1)  # [B, T]

        # Process each expert
        output = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (expert_idx == e)  # [B, T]
            if not mask.any():
                continue
            # Gather tokens for this expert
            x_e = x[mask]  # [N_e, D]
            # Forward through expert
            h = F.leaky_relu(F.linear(x_e, up_ws[e].to(x.dtype)), negative_slope=0.5)
            y_e = F.linear(h.square(), down_ws[e].to(x.dtype))
            # Scatter back
            weight_e = top1_weight[mask].unsqueeze(-1)  # [N_e, 1]
            output[mask] = y_e * weight_e

        return output
```

### 3.2 Bank Restructuring for MoE

```python
# Replace single MLP banks with per-expert banks
# Old: mlp_up_bank [num_layers, mlp_dim, model_dim]
# New: mlp_up_bank [num_experts * num_layers, expert_dim, model_dim]

num_experts = 2
n = num_unique_blocks  # 4
expert_dim = int(mlp_mult * model_dim) // num_experts

self.mlp_up_bank = nn.Parameter(
    torch.empty(num_experts * n, expert_dim, model_dim)
)
self.mlp_down_bank = nn.Parameter(
    torch.empty(num_experts * n, model_dim, expert_dim)
)
```

### 3.3 Load Balancing Loss

Add auxiliary load-balancing loss to prevent expert collapse:

```python
def load_balance_loss(router_logits: Tensor, expert_idx: Tensor,
                      num_experts: int) -> Tensor:
    """Encourage balanced expert utilization."""
    # Fraction of tokens routed to each expert
    density = torch.zeros(num_experts, device=expert_idx.device)
    for e in range(num_experts):
        density[e] = (expert_idx == e).float().mean()
    # Mean probability assigned to each expert
    probs = F.softmax(router_logits, dim=-1).mean(dim=(0, 1))
    # Switch loss: density * probability should be uniform
    return num_experts * (density * probs).sum()
```

### 3.4 Parameter Budget Verification

```
Per unique block MLP (2 experts, half-width each):
  2 × (expert_dim × dim + dim × expert_dim)
  = 2 × (1152 × 768 + 768 × 1152)   [mlp_mult=3.0, so expert_dim = 3*768/2 = 1152]
  = 2 × 1,769,472
  = 3,538,944

vs. single MLP (same total width):
  (2304 × 768 + 768 × 2304) = 3,538,944

Same param count! The router adds only 768 × 2 = 1,536 params.
MoE gives 2× capacity for free in terms of parameters.
```

### 3.5 Validation

- [ ] MoE forward pass matches expected output shape
- [ ] Both experts receive tokens (no collapse) — monitor routing distribution
- [ ] Load balance loss is < 0.1 after convergence
- [ ] Step time overhead < 10% vs. single MLP (gather/scatter cost)
- [ ] val_bpb improves over single MLP

---

## Phase 4: Fisher-Weighted Mixed Quantization (Days 9-10)

### 4.1 Fisher Information Computation

After training, compute per-parameter Fisher information to guide bit allocation:

```python
def compute_fisher_information(model, val_tokens, device, num_samples=500):
    """Compute diagonal Fisher information for each parameter tensor."""
    fisher = {name: torch.zeros_like(p, dtype=torch.float64)
              for name, p in model.named_parameters()}

    model.eval()
    seq_len = 2048
    for i in range(num_samples):
        start = random.randint(0, val_tokens.numel() - seq_len - 2)
        x = val_tokens[start:start+seq_len].unsqueeze(0).to(device, dtype=torch.int64)
        y = val_tokens[start+1:start+seq_len+1].unsqueeze(0).to(device, dtype=torch.int64)

        model.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.double().square()

    for name in fisher:
        fisher[name] /= num_samples

    return fisher
```

### 4.2 Bit Allocation Strategy

```python
def allocate_bits(fisher: dict, target_bytes: int, param_sizes: dict):
    """
    Allocate bits per tensor to minimize total quantization error
    weighted by Fisher information.

    Available bitwidths: {4, 5, 6, 8, 16}
    """
    # Compute Fisher sensitivity per tensor
    sensitivities = {}
    for name, f in fisher.items():
        # Average Fisher diagonal × number of params
        sensitivities[name] = f.mean().item() * param_sizes[name]

    # Sort by sensitivity descending
    sorted_params = sorted(sensitivities.items(), key=lambda x: -x[1])

    # Greedy allocation: assign highest bits to most sensitive params
    allocation = {}
    remaining_bytes = target_bytes

    for name, sensitivity in sorted_params:
        n_params = param_sizes[name]
        # Try each bitwidth from highest to lowest
        for bits in [16, 8, 6, 5, 4]:
            cost = n_params * bits / 8
            if cost <= remaining_bytes * 0.4:  # Don't let any single tensor take >40%
                allocation[name] = bits
                remaining_bytes -= cost
                break
        else:
            allocation[name] = 4  # Minimum
            remaining_bytes -= n_params * 4 / 8

    return allocation
```

### 4.3 Multi-Bitwidth Quantization Functions

```python
def quantize_per_row(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    """Quantize tensor with specified bitwidth."""
    clip_range = (1 << (bits - 1)) - 1  # e.g., 4-bit → 7, 6-bit → 31, 8-bit → 127

    t32 = t.float()
    if t32.ndim < 2:
        amax = t32.abs().max().item()
        scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
        return q, scale

    # Per-row quantization with optimal clipping
    best_q, best_s, best_err = None, None, float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
        recon = q.float() * s.float()[:, None]
        err = (t32 - recon).pow(2).mean().item()
        if err < best_err:
            best_q, best_s, best_err = q, s, err

    return best_q, best_s
```

### 4.4 Expected Size Budget

```
Component              | Params   | Bits | Bytes
-----------------------|----------|------|-------
Embedding (8192×768)   | 6.29M    | 6    | 4.72M
QO bank (8×768×768)    | 4.72M    | 6    | 3.54M
KV bank (8×256×768)    | 1.57M    | 6    | 1.18M
MLP up bank (8×1152×768)| 7.08M   | 4    | 3.54M  ← lower sensitivity
MLP down bank (8×768×1152)| 7.08M | 4    | 3.54M  ← lower sensitivity
Router weights          | 6.1K    | 16   | 12.2K
Adapters (32×~12K)     | 384K    | 8    | 384K
Control tensors         | ~50K    | 32   | 200K
-----------------------|----------|------|-------
Total weights           |         |      | ~13.5M
LZMA compression (~75%) |         |      | ~10.1M
Code                    |         |      | ~60K
-----------------------|----------|------|-------
Total artifact          |         |      | ~10.2M  ✓ Under 16MB
```

### 4.5 Validation

- [ ] Quantized model roundtrips correctly (dequant → eval matches)
- [ ] val_bpb degradation from quantization < 0.005
- [ ] Total artifact size < 15MB (leave 1MB headroom)
- [ ] Fisher allocation assigns more bits to attention than MLP

---

## Phase 5: Advanced TTT & Eval (Days 11-13)

### 5.1 LoRA-Based TTT

Replace full-parameter TTT with LoRA for faster, more stable adaptation:

```python
class LoRALayer(nn.Module):
    """Low-rank adapter for test-time training."""
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, base_weight: Tensor) -> Tensor:
        """Apply base_weight + LoRA adaptation."""
        # W_eff = W_base + scale * B @ A
        adapted = base_weight + self.scale * (self.B @ self.A)
        return F.linear(x, adapted)


def create_ttt_loras(model, rank=4):
    """Create LoRA adapters for all bank weights."""
    loras = {}
    for name in ['qo_bank', 'kv_bank', 'mlp_up_bank', 'mlp_down_bank']:
        bank = getattr(model, name)
        for i in range(bank.shape[0]):
            key = f"{name}.{i}"
            loras[key] = LoRALayer(bank.shape[2], bank.shape[1], rank=rank)
    return nn.ModuleDict(loras)


def eval_val_ttt_lora(
    args, base_model, rank, world_size, device,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride, log0=print,
):
    """Score-first TTT with LoRA adapters (legal evaluation)."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    # Create fresh LoRA adapters
    loras = create_ttt_loras(base_model, rank=4).to(device)
    optimizer = torch.optim.Adam(loras.parameters(), lr=args.ttt_lr, weight_decay=0.0)

    # Chunk-level processing: score first, then train
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    for ci in range(num_chunks):
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

        # Phase 1: SCORE (inference mode, LoRAs applied but frozen)
        # ... standard sliding window scoring with LoRA-augmented forward ...

        # Phase 2: TRAIN LoRAs on scored chunk (legal: already graded)
        if ci < num_chunks - 1 and args.ttt_epochs > 0:
            # Cosine LR schedule across chunks
            cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
            for pg in optimizer.param_groups:
                pg['lr'] = cos_lr

            # Train LoRAs on this chunk
            base_model.train()
            for _ep in range(args.ttt_epochs):
                # ... forward with LoRA-augmented weights ...
                # ... backward through LoRA params only ...
                optimizer.step()

    return val_loss, val_bpb
```

### 5.2 Adaptive TTT Learning Rate

```python
def adaptive_ttt_lr(chunk_loss: float, baseline_loss: float, base_lr: float) -> float:
    """Scale TTT learning rate based on chunk difficulty.

    High-loss chunks (unusual text) benefit from more aggressive adaptation.
    Low-loss chunks (predictable text) need less adaptation to avoid overfitting.
    """
    ratio = chunk_loss / max(baseline_loss, 1e-6)
    # Clamp ratio to [0.5, 3.0] range
    ratio = max(0.5, min(3.0, ratio))
    return base_lr * ratio
```

### 5.3 Sliding Window Improvements

```python
# Current: stride=64 with seq_len=2048
# Proposed: stride=32 for finer-grained context

# Additionally: use overlapping evaluation batches to maximize context
EVAL_STRIDE = 32  # Halve stride for more context overlap
EVAL_SEQ_LEN = 4096  # Double eval sequence length (if memory allows)
```

### 5.4 MC Dropout Ensemble at Eval Time

```python
def eval_with_mc_dropout(model, x, num_samples=3, dropout_p=0.05):
    """Average logits across multiple stochastic forward passes."""
    model.train()  # Enable dropout
    logits_sum = None
    for _ in range(num_samples):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward_logits(x)
        if logits_sum is None:
            logits_sum = logits.float()
        else:
            logits_sum += logits.float()
    return logits_sum / num_samples
```

**Note**: This requires adding dropout layers to the model (currently none exist).
Add `nn.Dropout(p=0.05)` after attention output and MLP output, only active during
MC-dropout eval (not during training or standard eval).

### 5.5 Validation

- [ ] LoRA TTT improves val_bpb over no-TTT baseline
- [ ] LoRA TTT is faster than full-parameter TTT (measure wall time)
- [ ] Adaptive LR improves over fixed LR on diverse text
- [ ] MC dropout ensemble gives consistent improvement
- [ ] Total eval time < 10 minutes on 8×H100

---

## Phase 6: Integration & Hyperparameter Tuning (Days 14-16)

### 6.1 Combined Architecture Summary

```python
# Final hyperparameters
class FinalHyperparameters:
    # Architecture
    vocab_size = 8192
    model_dim = 768
    num_heads = 12
    num_kv_heads = 4
    mlp_mult = 3.0
    num_unique_blocks = 4
    num_loops = 8
    num_experts = 2
    adapter_rank = 8
    tie_embeddings = True

    # Training
    train_seq_len = 2048
    train_batch_tokens = 786_432
    iterations = 20000
    warmup_steps = 20
    warmdown_iters = 3500
    max_wallclock_seconds = 600.0

    # Optimizer
    matrix_lr = 0.025
    scalar_lr = 0.025
    tied_embed_lr = 0.02
    muon_momentum = 0.99
    muon_backend_steps = 5
    grad_clip_norm = 0.3

    # Quantization
    quant_strategy = "fisher_mixed"  # int4 for MLP, int6 for attn, int8 for embed

    # TTT
    ttt_enabled = True
    ttt_lr = 0.002
    ttt_epochs = 3
    ttt_lora_rank = 4
    ttt_chunk_tokens = 32768

    # Eval
    eval_stride = 32
    eval_seq_len = 4096
    mc_dropout_samples = 3
```

### 6.2 Hyperparameter Sweep Plan

Run on 1×H100 with reduced iterations for fast iteration:

```
Sweep 1: Model dimension
  dim ∈ {640, 704, 768, 832}
  Fix: 4 blocks × 8 loops, 8K vocab

Sweep 2: Depth vs width tradeoff
  (3 blocks × 10 loops) vs (4 blocks × 8 loops) vs (5 blocks × 6 loops)
  Fix: dim=768, 8K vocab

Sweep 3: Number of experts
  experts ∈ {1, 2, 4}
  Fix: best from sweeps 1-2

Sweep 4: Learning rate grid
  matrix_lr ∈ {0.015, 0.020, 0.025, 0.030}
  embed_lr ∈ {0.010, 0.015, 0.020, 0.025}

Sweep 5: TTT hyperparameters
  ttt_lr ∈ {0.001, 0.002, 0.005}
  ttt_lora_rank ∈ {2, 4, 8}
  ttt_epochs ∈ {1, 2, 3, 5}

Sweep 6: Quantization
  Test all Fisher bit allocations vs uniform int6
```

### 6.3 Progressive Training Schedule

```python
def get_seq_len_schedule(step: int, total_steps: int) -> int:
    """Progressive sequence length curriculum."""
    frac = step / total_steps
    if frac < 0.3:
        return 512   # Short seqs for fast early training
    elif frac < 0.6:
        return 1024  # Medium seqs for mid-training
    else:
        return 2048  # Full length for final training
```

### 6.4 Final Submission Checklist

- [ ] 3 training runs with different seeds, all under 10 min on 8×H100
- [ ] Mean val_bpb with std reported
- [ ] Statistical significance: p < 0.01 vs current SOTA (1.1194)
- [ ] Artifact size < 16,000,000 bytes (code + compressed model)
- [ ] Eval time < 10 min on 8×H100
- [ ] No external downloads during eval
- [ ] `train_gpt.py` compiles and runs from records/ folder
- [ ] README.md with full description
- [ ] submission.json with metadata
- [ ] Train logs attached

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|------------|------------|
| Depth recurrence too slow per step | High | Medium | Reduce loops to 6, or use torch.compile tricks |
| 8K vocab embedding too large | High | Low | Fall back to 4K vocab, or use int4 for embedding |
| MoE expert collapse | Medium | Medium | Auxiliary load balance loss, router temp scheduling |
| Fisher computation too expensive | Low | Low | Use only 100 samples, or skip and use heuristic allocation |
| torch.compile incompatible with loops | High | Medium | Unroll loops manually, or compile individual blocks |
| Total artifact > 16MB | High | Low | Reduce dim to 704, reduce adapter rank, more aggressive quant |
| TTT eval exceeds 10 min | Medium | Medium | Reduce ttt_epochs, increase chunk size, skip MC dropout |
| Gradient instability at 32 effective depth | Medium | Medium | Gradient clipping per loop, skip connection normalization |

---

## Timeline Summary

```
Day 1:      Phase 0 — Baseline reproduction, instrumentation
Days 2-4:   Phase 1 — Depth recurrence implementation + validation
Days 5-6:   Phase 2 — 8K vocabulary tokenizer + data pipeline
Days 7-8:   Phase 3 — MoE MLP implementation
Days 9-10:  Phase 4 — Fisher-weighted quantization
Days 11-13: Phase 5 — Advanced TTT + eval improvements
Days 14-16: Phase 6 — Integration, hyperparameter tuning, submission
```

Each phase has clear validation checkpoints. If a phase fails to improve val_bpb,
skip it and move to the next. The phases are designed to be independently valuable.

---

## Appendix A: Key Code Modifications Map

```
train_gpt.py modifications:

Line ~400: class GPT → class RecurrentGPT
  - Add LoopAdapter class
  - Restructure __init__ for 4 unique blocks
  - Rewrite forward() with loop logic

Line ~300: class MLP → class MoEMLP
  - Add router
  - Add per-expert forward with gather/scatter
  - Add load balance loss

Line ~100: class Muon
  - Adjust bank sizes for 4 unique blocks × 2 experts

Line ~650: main()
  - Add Fisher computation after training
  - Replace mixed_quantize_int6 with fisher_mixed_quantize
  - Add LoRA TTT eval path

New files:
  - retokenize.py (data pipeline for 8K vocab)
  - fineweb_8192_bpe.model (tokenizer)
```

## Appendix B: Theoretical Analysis

### Why Depth Recurrence Works at This Scale

The information bottleneck principle suggests that at small parameter counts,
**reusing parameters across depth gives better compression** than having many
unique but shallow layers. Each parameter in a recurrent block learns a
**general-purpose transformation** that's applied in context-dependent ways
via the adapters.

Empirically, Universal Transformers (Dehghani et al., 2019) showed that
weight-tied transformers match or exceed standard transformers at the same
parameter count when depth exceeds width. Our regime (32 effective layers,
768 dim) is exactly where this crossover occurs.

### Why MoE Helps Under Parameter Constraints

MoE is not about having more parameters — with 2 half-width experts,
total params equal a single full-width MLP. The benefit is **conditional
computation**: different tokens activate different experts, so the model
learns **specialized sub-networks** for different token types (function words
vs content words, common vs rare tokens, etc.) without parameter interference.

This is especially powerful with depth recurrence: at different loop iterations,
the same MoE layer routes different tokens to different experts, effectively
creating iteration-dependent MLP behavior without extra parameters.

### Why 8K Vocab Improves BPB

The val_bpb metric is:
  bpb = (cross_entropy_loss / ln(2)) × (tokens / bytes)

With 8K vocab vs 1K vocab:
- tokens/bytes drops from ~1.8 to ~1.3 (27% reduction)
- cross_entropy increases (more classes) but by less than 27%
- Net: bpb improves by ~0.02-0.03 based on empirical data from
  the ternary quantization submission which already uses 8K vocab