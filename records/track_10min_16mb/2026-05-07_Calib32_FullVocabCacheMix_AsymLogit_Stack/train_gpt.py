from __future__ import annotations
import copy
import ctypes
import glob
import heapq
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from flash_attn_interface import flash_attn_func as flash_attn_3_func
TORCH_COMPILE_DISABLE = bool(int(os.environ.get("TORCH_COMPILE_DISABLE", "0")))
TORCH_COMPILE_DYNAMIC = bool(int(os.environ.get("TORCH_COMPILE_DYNAMIC", "0")))
TORCH_COMPILE_FULLGRAPH = bool(int(os.environ.get("TORCH_COMPILE_FULLGRAPH", "1")))
TORCH_COMPILE_BACKEND = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
TORCH_COMPILE_MODE = os.environ.get("TORCH_COMPILE_MODE") or None
TORCH_COMPILE_EVAL_MODE = os.environ.get("TORCH_COMPILE_EVAL_MODE") or None
TORCH_COMPILE_EVAL_DISABLE = bool(int(os.environ.get(
    "TORCH_COMPILE_EVAL_DISABLE",
    "1" if (TORCH_COMPILE_MODE or "").startswith("max-autotune") else "0",
)))
def cudagraph_step_begin() -> None:
    if TORCH_COMPILE_DISABLE or TORCH_COMPILE_BACKEND != "inductor":
        return
    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
def compile_with_env(fn, *, is_eval: bool = False):
    if TORCH_COMPILE_DISABLE:
        return fn
    if is_eval and TORCH_COMPILE_EVAL_DISABLE:
        return fn
    kwargs = {
        "dynamic": TORCH_COMPILE_DYNAMIC,
        "fullgraph": TORCH_COMPILE_FULLGRAPH,
        "backend": TORCH_COMPILE_BACKEND,
    }
    compile_mode = TORCH_COMPILE_EVAL_MODE if is_eval and TORCH_COMPILE_EVAL_MODE is not None else TORCH_COMPILE_MODE
    if compile_mode is not None:
        kwargs["mode"] = compile_mode
    return torch.compile(fn, **kwargs)

def submission_code_bytes() -> int:
    total = len(Path(__file__).resolve().read_bytes())
    helper = Path(__file__).with_name("online_full_vocab_cache.c")
    if helper.exists():
        total += len(helper.read_bytes())
    return total

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    lr_schedule_reference_step_ms = float(os.environ.get("LR_SCHEDULE_REFERENCE_STEP_MS", "120"))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    asym_logit_rescale = bool(int(os.environ.get("ASYM_LOGIT_RESCALE", "1")))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.04))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.03))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.03))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.5))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    swa_start_step = int(os.environ.get("SWA_START_STEP", "0"))
    swa_start_scale = float(os.environ.get("SWA_START_SCALE", "0.4"))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))
    bigram_hash_mode = os.environ.get("BIGRAM_HASH_MODE", "gf13").strip().lower()
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "7,8,9,10")
    attention_sink = bool(int(os.environ.get("ATTENTION_SINK", "1")))
    expert_mixer_enabled = bool(int(os.environ.get("EXPERT_MIXER_ENABLED", "0")))
    expert_recent_window = int(os.environ.get("EXPERT_RECENT_WINDOW", "32"))
    expert_cache_window = int(os.environ.get("EXPERT_CACHE_WINDOW", "64"))
    expert_cache_dim = int(os.environ.get("EXPERT_CACHE_DIM", "64"))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_online_agree_order = os.environ.get("TTT_ONLINE_AGREE_ORDER", "ttt_first").strip().lower()
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    negative_slope = float(os.environ.get("NEGATIVE_SLOPE", 0.5))
    use_gptq = bool(int(os.environ.get("USE_GPTQ", "1")))
    gptq_calib_samples = int(os.environ.get("GPTQ_CALIB_SAMPLES", "32"))
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", "64"))
    gptq_percdamp = float(os.environ.get("GPTQ_PERCDAMP", "0.01"))
    int8_attn_layers = os.environ.get("INT8_ATTN_LAYERS", "0,-1")
    int8_mlp_layers = os.environ.get("INT8_MLP_LAYERS", "")
    gptq_reserve_ms = float(os.environ.get("GPTQ_RESERVE_MS", "4500"))
    quant_clip_range = int(os.environ.get("QUANT_CLIP_RANGE", 31))
    lqer_enabled = bool(int(os.environ.get("LQER_ENABLED", "1")))
    lqer_rank = int(os.environ.get("LQER_RANK", "4"))
    lqer_top_k = int(os.environ.get("LQER_TOP_K", "3"))
    lqer_asym_group = int(os.environ.get("LQER_ASYM_GROUP", "32"))
    ema_start_decay = float(os.environ.get("EMA_START_DECAY", "0.99"))
    ema_end_decay = float(os.environ.get("EMA_END_DECAY", "0.999"))

def parse_layer_spec(spec: str, num_layers: int) -> set[int]:
    selected: set[int] = set()
    if not spec:
        return selected
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "first":
            idx = 0
        elif token == "last":
            idx = num_layers - 1
        else:
            try:
                idx = int(token)
            except ValueError:
                continue
            if idx < 0:
                idx += num_layers
        if 0 <= idx < num_layers:
            selected.add(idx)
    return selected

def layer_idx_from_name(name: str) -> int | None:
    if not name.startswith("blocks."):
        return None
    parts = name.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None

# --- Batched Newton-Schulz orthogonalization ---

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Batched Newton-Schulz orthogonalization. G: (B,M,N) or (M,N)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X

# --- Parallel Muon optimizer ---

class Muon(torch.optim.Optimizer):
    """Parallel Muon: post-backward reduce-scatter -> local NS5 -> all-gather.

    No DDP for bank params. After backward, this optimizer:
    1. Launches async reduce-scatter for all banks (biggest first)
    2. Returns control so Adam can step on small params while RS is in-flight
    3. Waits for each RS, runs local NS5 on the shard, launches async all-gather
    4. Each all-gather overlaps with next bank's NS5
    """
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
        self._built = False

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size

        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    'p': p,
                    'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True

    def launch_reduce_scatters(self):
        """Phase 1: launch async reduce-scatter for all banks. Call right after backward."""
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m['padded_grad']
            pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']:
                pg[m['B']:].zero_()
            fut = dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        """Phase 3: wait for RS, local NS5, all-gather. Call AFTER Adam steps."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self._built:
            self._build()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)

            prev_ag_handle = None
            prev_m = None

            sharded = self._distributed and hasattr(self, '_rs_futures')

            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None:
                    continue

                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m['p']
                    upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m['shard']
                    buf = m['shard_mom']
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                update = zeropower_via_newtonschulz5(update, steps=backend_steps)

                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m['full_update'], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])

            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m['p']
                upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

            if hasattr(self, '_rs_futures'):
                del self._rs_futures

        return loss

# --- Tokenizer evaluation helpers ---

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )
def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]
def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    prime_rotary_caches(model, device, seq_len)
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                cudagraph_step_begin()
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# --- Quantization helpers ---

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,vr_lambda",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())
def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t
def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale
def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats
def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

# --- Data loading ---

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}

def _read_num_tokens(file: Path) -> int:
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n

def _get_shard_memmap(file: Path) -> np.memmap:
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm

class DistributedTokenLoader:
    """Coprime-stride + multi-shard loader (PR #726 style). No daemon thread / prefetch."""
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self._num_tokens = np.array([_read_num_tokens(f) for f in self.files], dtype=np.int64)
        seed = 0
        for f in self.files:
            for b in str(f).encode():
                seed = ((seed ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._cfg: tuple[int, int, int, int] | None = None
        self._eligible_shards: np.ndarray | None = None
        self._base_block_counts: np.ndarray | None = None
        n = len(self.files)
        self._cursor_phase = np.zeros(n, dtype=np.int64)
        self._cursor_block_count = np.zeros(n, dtype=np.int64)
        self._cursor_next = np.zeros(n, dtype=np.int64)
        self._cursor_start = np.zeros(n, dtype=np.int64)
        self._cursor_stride = np.ones(n, dtype=np.int64)
        self._cursor_init = np.zeros(n, dtype=np.bool_)
        self._batches_built = 0
    def _pick_coprime_stride(self, n: int) -> int:
        if n <= 1:
            return 1
        while True:
            s = int(self._rng.integers(1, n))
            if math.gcd(s, n) == 1:
                return s
    def _reset_cursor(self, si: int, seq_len: int) -> None:
        nt = int(self._num_tokens[si])
        max_phase = min(seq_len - 1, max(0, nt - seq_len - 1))
        phase = int(self._rng.integers(max_phase + 1)) if max_phase > 0 else 0
        bc = (nt - 1 - phase) // seq_len
        self._cursor_phase[si] = phase
        self._cursor_block_count[si] = bc
        self._cursor_next[si] = 0
        self._cursor_start[si] = int(self._rng.integers(bc)) if bc > 1 else 0
        self._cursor_stride[si] = self._pick_coprime_stride(bc)
        self._cursor_init[si] = True
    def _ensure_cursor(self, si: int, seq_len: int) -> None:
        if not self._cursor_init[si] or self._cursor_next[si] >= self._cursor_block_count[si]:
            self._reset_cursor(si, seq_len)
    def _take_from_shard(self, si: int, seq_len: int, count: int, out: list[tuple[int, int]]) -> None:
        rem = count
        while rem > 0:
            self._ensure_cursor(si, seq_len)
            bc = int(self._cursor_block_count[si])
            ni = int(self._cursor_next[si])
            take = min(rem, bc - ni)
            phase = int(self._cursor_phase[si])
            start = int(self._cursor_start[si])
            stride = int(self._cursor_stride[si])
            for j in range(take):
                bi = (start + (ni + j) * stride) % bc
                out.append((si, phase + bi * seq_len))
            self._cursor_next[si] = ni + take
            rem -= take
    def _init_pipeline(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> None:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        num_seqs = local_tokens // seq_len
        global_num_seqs = num_seqs * self.world_size
        self._cfg = (local_tokens, seq_len, num_seqs, global_num_seqs)
        bbc = (self._num_tokens - 1) // seq_len
        eligible = bbc > 0
        self._eligible_shards = np.nonzero(eligible)[0].astype(np.int64)
        self._base_block_counts = bbc[self._eligible_shards].astype(np.int64)
    def _sample_global_windows(self) -> list[tuple[int, int]]:
        assert self._cfg is not None and self._eligible_shards is not None
        _, seq_len, _, gns = self._cfg
        ec = int(self._eligible_shards.size)
        progress = min(self._batches_built / 1800.0, 1.0)
        remaining = np.empty(ec, dtype=np.float64)
        for i, si in enumerate(self._eligible_shards.tolist()):
            if self._cursor_init[si]:
                r = int(self._cursor_block_count[si]) - int(self._cursor_next[si])
                remaining[i] = float(max(r, 1))
            else:
                remaining[i] = float(self._base_block_counts[i])
        alpha = 0.90 - 0.40 * progress
        weights = np.power(remaining, alpha)
        ws = float(weights.sum())
        if not np.isfinite(ws) or ws <= 0.0:
            weights = np.ones(ec, dtype=np.float64)
            ws = float(weights.sum())
        probs = weights / ws
        low = min(max(8, self.world_size), ec, gns)
        high = min(max(32, self.world_size * 8), ec, gns)
        mix = max(1, min(int(round(low + progress * (high - low))), ec, gns))
        cp = self._rng.choice(ec, size=mix, replace=False, p=probs)
        cs = self._eligible_shards[cp]
        cpr = probs[cp].copy()
        cpr /= cpr.sum()
        counts = np.ones(mix, dtype=np.int64)
        extra = gns - mix
        if extra > 0:
            counts += self._rng.multinomial(extra, cpr).astype(np.int64)
        perm = self._rng.permutation(mix)
        cs, counts = cs[perm], counts[perm]
        buckets: list[list[tuple[int, int]]] = []
        for si, cnt in zip(cs.tolist(), counts.tolist()):
            b: list[tuple[int, int]] = []
            self._take_from_shard(int(si), seq_len, int(cnt), b)
            if b:
                if len(b) > 1:
                    bp = self._rng.permutation(len(b))
                    b = [b[int(k)] for k in bp.tolist()]
                buckets.append(b)
        windows: list[tuple[int, int]] = []
        active = [i for i, bk in enumerate(buckets) if bk]
        while active:
            order = self._rng.permutation(len(active))
            new_active: list[int] = []
            for oi in order.tolist():
                bi = active[oi]
                if buckets[bi]:
                    windows.append(buckets[bi].pop())
                if buckets[bi]:
                    new_active.append(bi)
            active = new_active
        return windows
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if self._cfg is None:
            self._init_pipeline(global_tokens, seq_len, grad_accum_steps)
        _, _, num_seqs, gns = self._cfg
        gw = self._sample_global_windows()
        local_w = gw[self.rank::self.world_size]
        x = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        y = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        for slot, (si, pos) in enumerate(local_w):
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[pos:pos + seq_len + 1], dtype=np.int64))
            x[slot] = window[:-1]
            y[slot] = window[1:]
        self._batches_built += 1
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# --- Transformer modules ---

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        rd = self.rope_dims
        if seq_len > self.train_seq_len:
            scale = seq_len / self.train_seq_len
            new_base = self.base * (scale ** (rd / (rd - 2)))
            inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
        else:
            inv_freq = self.inv_freq.to(device)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        self._cos_cached = freqs.cos()[None, :, None, :].contiguous()
        self._sin_cached = freqs.sin()[None, :, None, :].contiguous()
        self._seq_len_cached = seq_len
    def prime_cache(self, seq_len: int, device: torch.device) -> None:
        with torch.no_grad():
            self._build_cache(seq_len, device)
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
        ):
            self.prime_cache(seq_len, device)
        cos = self._cos_cached[:, :seq_len]
        sin = self._sin_cached[:, :seq_len]
        if cos.dtype != dtype:
            cos = cos.to(dtype=dtype)
        if sin.dtype != dtype:
            sin = sin.to(dtype=dtype)
        return cos, sin
def prime_rotary_caches(module: nn.Module, device: torch.device, *seq_lens: int) -> None:
    raw_module = getattr(module, "_orig_mod", module)
    max_seq_len = max((int(seq_len) for seq_len in seq_lens if int(seq_len) > 0), default=0)
    if max_seq_len <= 0:
        return
    for submodule in raw_module.modules():
        if isinstance(submodule, Rotary):
            submodule.prime_cache(max_seq_len, device)
def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False  # set by GPT.__init__ for deep layers only
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Efficient XSA: subtract self-value projection via GQA-aware reshape (no repeat_interleave).
        y: [B, T, H, D], v: [B, T, Hkv, D]. H must be divisible by Hkv."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)        # [B, T, Hkv, group, D]
        vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D] -- broadcast ready
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(
        self,
        x: Tensor,
        q_w: Tensor,
        k_w: Tensor,
        v_w: Tensor,
        out_w: Tensor,
        v_embed: Tensor | None = None,
        sink_embed: Tensor | None = None,
    ) -> Tensor:
        if getattr(self, '_save_gptq', False):
            self._gptq_qkv_in = x.detach()
        bsz, seqlen, dim = x.shape
        q_w = q_w.to(x.dtype)
        k_w = k_w.to(x.dtype)
        v_w = v_w.to(x.dtype)
        out_w = out_w.to(x.dtype)
        q = F.linear(x, q_w).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        if sink_embed is not None:
            sink_x = sink_embed.to(dtype=x.dtype)[None, None, :].expand(bsz, 1, dim)
            sink_q = F.linear(sink_x, q_w).reshape(bsz, 1, self.num_heads, self.head_dim)
            sink_k = F.linear(sink_x, k_w).reshape(bsz, 1, self.num_kv_heads, self.head_dim)
            sink_v = F.linear(sink_x, v_w).reshape(bsz, 1, self.num_kv_heads, self.head_dim)
            sink_q = F.rms_norm(sink_q, (sink_q.size(-1),))
            sink_k = F.rms_norm(sink_k, (sink_k.size(-1),))
            q = torch.cat((sink_q, q), dim=1)
            k = torch.cat((sink_k, k), dim=1)
            v = torch.cat((sink_v, v), dim=1)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if sink_embed is not None:
            y = y[:, 1:]
        y = y.reshape(bsz, seqlen, dim)
        if getattr(self, '_save_gptq', False):
            self._gptq_o_in = y.detach()
        return F.linear(y, out_w)

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

def _build_gf13_hash_tables(mod: int) -> tuple[np.ndarray, np.ndarray]:
    field_bits = 13
    field_size = 1 << field_bits
    mask = field_size - 1
    primitive_poly = 0x201B  # x^13 + x^4 + x^3 + x + 1, period 8191.
    exp = np.empty(field_size - 1, dtype=np.int32)
    log = np.zeros(field_size, dtype=np.int32)
    x = 1
    for i in range(field_size - 1):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & field_size:
            x ^= primitive_poly
        x &= mask
    mix = np.zeros(field_size, dtype=np.int64)
    mix_power = 137
    for a in range(1, field_size):
        mix[a] = int(exp[(int(log[a]) + mix_power) % (field_size - 1)])
    hash_lut = np.full(field_size, max(mod, 0), dtype=np.int64)
    if mod > 0:
        hash_lut[1:] = (log[1:] % mod).astype(np.int64)
    return mix, hash_lut

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int, hash_mode: str = "gf13"):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.hash_mode = hash_mode
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        if self.hash_mode == "gf13":
            mix_lut, hash_lut = _build_gf13_hash_tables(bigram_vocab_size - 1)
            self.register_buffer("gf13_mix_lut", torch.tensor(mix_lut, dtype=torch.long), persistent=False)
            self.register_buffer("gf13_hash_lut", torch.tensor(hash_lut, dtype=torch.long), persistent=False)
    def _legacy_bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def _gf13_bigram_hash(self, tokens: Tensor) -> Tensor:
        t = torch.bitwise_and(tokens.to(torch.long), 8191)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        mixed_prev = self.gf13_mix_lut[t[..., :-1]]
        field_elem = torch.bitwise_xor(t[..., 1:], mixed_prev)
        out[..., 1:] = self.gf13_hash_lut[field_elem]
        return out
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        if self.hash_mode == "gf13":
            return self._gf13_bigram_hash(tokens)
        return self._legacy_bigram_hash(tokens)
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers.
    Each table maps vocab tokens to a low-dim embedding, projected to model_dim."""
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, neg_slope: float = 0.5):
        super().__init__()
        self.neg_slope = neg_slope
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        if getattr(self, '_save_gptq', False):
            self._gptq_up_in = x.detach()
        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=self.neg_slope)
        x2 = x.square()
        if getattr(self, '_save_gptq', False):
            self._gptq_down_in = x2.detach()
        return F.linear(x2, down_w.to(x.dtype))

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
        ln_scale: bool = False,
        neg_slope: float = 0.5,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult, neg_slope=neg_slope)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        q_w: Tensor,
        k_w: Tensor,
        v_w: Tensor,
        out_w: Tensor,
        up_w: Tensor,
        down_w: Tensor,
        v_embed: Tensor | None = None,
        sink_embed: Tensor | None = None,
    ) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor,
            q_w,
            k_w,
            v_w,
            out_w,
            v_embed=v_embed,
            sink_embed=sink_embed,
        )
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        mlp_out = self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out + mlp_out

class RecentTokenReplayExpert(nn.Module):
    """Bias toward tokens that followed recent repeats of the current token."""
    def __init__(self, dim: int, window: int):
        super().__init__()
        self.window = max(window, 0)
        self.gate = CastedLinear(dim, 1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    def forward(self, hidden: Tensor, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seqlen, _ = hidden.shape
        out_token = input_ids.new_zeros((bsz, seqlen))
        out_bias = hidden.new_zeros((bsz, seqlen))
        if seqlen < 2 or self.window <= 0:
            return out_token, out_bias
        best_score = hidden.new_full((bsz, seqlen), -1e4)
        max_lag = min(self.window, seqlen - 1)
        recency_den = float(max(max_lag - 1, 1))
        for lag in range(1, max_lag + 1):
            match = input_ids[:, lag:] == input_ids[:, :-lag]
            score = match.to(dtype=hidden.dtype)
            score = score + (max_lag - lag) / recency_den * 0.1
            score = torch.where(match, score, hidden.new_full(score.shape, -1e4))
            better = score > best_score[:, lag:]
            best_score[:, lag:] = torch.where(better, score, best_score[:, lag:])
            cand = input_ids[:, 1:seqlen - lag + 1]
            out_token[:, lag:] = torch.where(better, cand, out_token[:, lag:])
        gate = torch.sigmoid(self.gate(hidden).squeeze(-1).float())
        out_bias = gate * (best_score > -1e3).to(dtype=gate.dtype) * F.softplus(self.scale.float())
        return out_token, out_bias.to(dtype=hidden.dtype)

class NeuralCacheExpert(nn.Module):
    """Small trainable cache over recent hidden states."""
    def __init__(self, dim: int, cache_dim: int, window: int):
        super().__init__()
        self.window = max(window, 0)
        self.query = CastedLinear(dim, cache_dim, bias=False)
        self.key = CastedLinear(dim, cache_dim, bias=False)
        self.gate = CastedLinear(dim, 1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.75, dtype=torch.float32))
        self.threshold = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.temperature = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
    def forward(self, hidden: Tensor, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seqlen, _ = hidden.shape
        out_token = input_ids.new_zeros((bsz, seqlen))
        out_bias = hidden.new_zeros((bsz, seqlen))
        if seqlen < 2 or self.window <= 0:
            return out_token, out_bias
        q = F.normalize(self.query(hidden), dim=-1, eps=1e-6)
        k = F.normalize(self.key(hidden), dim=-1, eps=1e-6)
        best_score = hidden.new_full((bsz, seqlen), -1e4)
        max_lag = min(self.window, seqlen - 1)
        recency_den = float(max(max_lag - 1, 1))
        for lag in range(1, max_lag + 1):
            score = (q[:, lag:] * k[:, :-lag]).sum(dim=-1)
            score = score + (max_lag - lag) / recency_den * 0.05
            better = score > best_score[:, lag:]
            best_score[:, lag:] = torch.where(better, score, best_score[:, lag:])
            cand = input_ids[:, 1:seqlen - lag + 1]
            out_token[:, lag:] = torch.where(better, cand, out_token[:, lag:])
        gate = torch.sigmoid(self.gate(hidden).squeeze(-1).float())
        sharpness = F.softplus(self.temperature.float())
        confidence = torch.sigmoid((best_score.float() - self.threshold.float()) * sharpness)
        out_bias = gate * confidence * F.softplus(self.scale.float())
        return out_token, out_bias.to(dtype=hidden.dtype)

class CausalExpertMixer(nn.Module):
    def __init__(self, dim: int, cache_dim: int, recent_window: int, cache_window: int):
        super().__init__()
        self.recent = RecentTokenReplayExpert(dim, recent_window) if recent_window > 0 else None
        self.cache = NeuralCacheExpert(dim, cache_dim, cache_window) if cache_window > 0 else None
    def forward(self, hidden: Tensor, input_ids: Tensor, logits: Tensor) -> Tensor:
        if self.recent is None and self.cache is None:
            return logits
        mixed = logits.clone()
        for expert in (self.recent, self.cache):
            if expert is None:
                continue
            top_token, bias = expert(hidden, input_ids)
            mixed.scatter_add_(-1, top_token.unsqueeze(-1), bias.unsqueeze(-1).to(dtype=mixed.dtype))
        return mixed

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        asym_logit_rescale: bool,
        rope_base: float,
        qk_gain_init: float,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 112,
        bigram_hash_mode: str = "gf13",
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "7,8,9,10",
        neg_slope: float = 0.5,
        attention_sink: bool = True,
        expert_mixer_enabled: bool = True,
        expert_recent_window: int = 32,
        expert_cache_window: int = 64,
        expert_cache_dim: int = 64,
    ):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)  # kv_dim for value projection
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.asym_logit_rescale = bool(asym_logit_rescale)
        if self.asym_logit_rescale:
            self.softcap_pos = nn.Parameter(torch.tensor(float(logit_softcap), dtype=torch.float32))
            self.softcap_neg = nn.Parameter(torch.tensor(float(logit_softcap), dtype=torch.float32))
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(
            bigram_vocab_size,
            bigram_dim,
            model_dim,
            hash_mode=bigram_hash_mode,
        ) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.attention_sink = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32)) if attention_sink else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)
        self.num_layers = num_layers
        self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                    ln_scale=ln_scale,
                    neg_slope=neg_slope,
                )
                for i in range(num_layers)
            ]
        )
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim_ve = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim_ve)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()  # keep empty for compat
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        self.expert_mixer = CausalExpertMixer(
            model_dim,
            expert_cache_dim,
            expert_recent_window,
            expert_cache_window,
        ) if expert_mixer_enabled else None
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.attention_sink is not None:
            nn.init.normal_(self.attention_sink, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)        # Q
            nn.init.zeros_(self.qo_bank.data[n + i])                    # Out (zero init)
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)        # K
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)    # V
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)    # MLP up
            nn.init.zeros_(self.mlp_down_bank.data[i])                  # MLP down (zero init)
            self.qo_bank.data[n + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        """Get value embedding for a specific layer using shared table + per-layer scale."""
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def _softcap_logits(self, logits_proj: Tensor) -> Tensor:
        if not self.asym_logit_rescale:
            return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        sp = self.softcap_pos.float().clamp_min(1e-3).to(dtype=logits_proj.dtype)
        sn = self.softcap_neg.float().clamp_min(1e-3).to(dtype=logits_proj.dtype)
        return torch.where(
            logits_proj >= 0,
            sp * torch.tanh(logits_proj / sp),
            sn * torch.tanh(logits_proj / sn),
        )
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        sink_embed = self.attention_sink
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0,
                self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                v_embed=ve, sink_embed=sink_embed)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                v_embed=ve, sink_embed=sink_embed)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self._softcap_logits(logits_proj)
        logits = logits.view(x.size(0), x.size(1), -1)
        if self.expert_mixer is not None:
            logits = self.expert_mixer(x, input_ids, logits)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets, reduction="mean")
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        sink_embed = self.attention_sink
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0,
                self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                v_embed=ve, sink_embed=sink_embed)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                v_embed=ve, sink_embed=sink_embed)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self._softcap_logits(logits_proj)
        if self.expert_mixer is not None:
            logits = self.expert_mixer(x, input_ids, logits)
        return logits

# --- Sliding window evaluation ---

def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    seq_len = eval_seq_len or args.train_seq_len
    prime_rotary_caches(base_model, device, seq_len)
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = compile_with_env(base_model.forward_logits, is_eval=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cudagraph_step_begin()
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte

def eval_val_sliding_ttt(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32, log0=print,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    prime_rotary_caches(base_model, device, seq_len)
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    nc = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(nc)]
    for ws in window_starts:
        s = 0 if ws == 0 else max(min(ws + seq_len, total_tokens) - ws - stride, 0)
        chunk_windows[min((ws + s) // ttt_chunk, nc - 1)].append(ws)
    log0(f"ttt_sliding:start chunks={nc} chunk_tokens={ttt_chunk} total_windows={len(window_starts)} stride={stride}")
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    frozen = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        if any(f"blocks.{bi}." in name for bi in frozen): p.requires_grad_(False)
        else: p.requires_grad_(True); ttt_params.append(p)
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()
    for ci in range(nc):
        windows = chunk_windows[ci]
        if not windows: continue
        cs, ce = ci * ttt_chunk, min((ci + 1) * ttt_chunk, total_tokens)
        ms, me = (len(windows) * rank) // world_size, (len(windows) * (rank + 1)) // world_size
        my_wins = windows[ms:me]
        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_wins), batch_seqs):
                bws = my_wins[bi:bi + batch_seqs]; bsz = len(bws)
                xb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                yb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(bws):
                    end = min(ws + seq_len, total_tokens); wlen = end - ws; wlens.append(wlen)
                    ct = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    xb[i, :wlen] = ct[:-1]; yb[i, :wlen] = ct[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(xb)
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), yb.reshape(-1), reduction="none").reshape(bsz, seq_len)
                for i, ws in enumerate(bws):
                    wlen = wlens[i]; s = 0 if ws == 0 else max(wlen - stride, 0)
                    loss_sum += nll[i, s:wlen].to(torch.float64).sum(); token_count += float(wlen - s)
                    tgt, prev = yb[i, s:wlen], xb[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()
        if ci < nc - 1 and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (ce - cs) // seq_len
            if chunk_seqs > 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(nc - 1, 1)))
                ms2 = (chunk_seqs * rank) // world_size; me2 = (chunk_seqs * (rank + 1)) // world_size
                for _ep in range(args.ttt_epochs):
                    for bs in range(0, me2 - ms2, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, me2 - ms2)
                        st = cs + (ms2 + bs) * seq_len; et = cs + (ms2 + be) * seq_len + 1
                        if et > val_tokens.numel(): continue
                        loc = val_tokens[st:et].to(device=device, dtype=torch.int64)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(loc[:-1].reshape(-1, seq_len), loc[1:].reshape(-1, seq_len))
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip); optimizer.step()
        if rank == 0 and (ci % 10 == 0 or ci == nc - 1):
            rl = loss_sum.item() / max(token_count.item(), 1)
            log0(f"  ttt_chunk [{ci+1}/{nc}] bpb={rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0:.6f}")
    if dist.is_available() and dist.is_initialized():
        for t in [loss_sum, token_count, byte_count]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    for p in base_model.parameters(): p.requires_grad_(True)
    base_model.eval()
    log0(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb

# --- GPTQ-lite int6 quantization ---

def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
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
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale

def _dequant_row_scaled(q: Tensor, s: Tensor) -> Tensor:
    if s.ndim > 0:
        return q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
    return q.float() * float(s.item())

def _lqer_residual_score(W: Tensor, q: Tensor, s: Tensor) -> float:
    residual = W.float() - _dequant_row_scaled(q, s)
    return float(residual.pow(2).sum().item())

def _lqer_group_correction(residual: Tensor, group: int) -> tuple[Tensor, Tensor | None]:
    if group <= 0 or residual.ndim != 2:
        return residual, None
    rows, cols = residual.shape
    groups = (cols + group - 1) // group
    pad = groups * group - cols
    if pad > 0:
        padded = F.pad(residual, (0, pad))
    else:
        padded = residual
    means = padded.reshape(rows, groups, group).mean(dim=2)
    correction = means.unsqueeze(-1).expand(rows, groups, group).reshape(rows, groups * group)[:, :cols]
    return residual - correction, means.to(torch.float16).contiguous()

def _apply_lqer_group(t: Tensor, means: Tensor, group: int) -> Tensor:
    if group <= 0:
        return t
    rows, cols = t.shape
    groups = means.shape[1]
    correction = means.float().unsqueeze(-1).expand(rows, groups, group).reshape(rows, groups * group)[:, :cols]
    return t + correction

def _lqer_factorize_residual(
    W: Tensor,
    q: Tensor,
    s: Tensor,
    *,
    rank: int,
    asym_group: int,
) -> dict[str, Tensor]:
    residual = W.float() - _dequant_row_scaled(q, s)
    residual, means = _lqer_group_correction(residual, asym_group)
    pieces: dict[str, Tensor] = {}
    if means is not None:
        pieces["mean"] = means
    r = min(max(int(rank), 0), residual.shape[0], residual.shape[1])
    if r <= 0:
        return pieces
    try:
        u, sv, vh = torch.linalg.svd(residual, full_matrices=False)
    except RuntimeError:
        return pieces
    a = u[:, :r] * sv[:r].unsqueeze(0)
    b = vh[:r, :]
    pieces["a"] = a.to(torch.float16).contiguous()
    pieces["b"] = b.to(torch.float16).contiguous()
    return pieces

def _unbank_state_dict(sd: dict[str, Tensor], num_layers: int) -> dict[str, Tensor]:
    """Convert 3D bank tensors into individual 2D tensors with standard names."""
    out: dict[str, Tensor] = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"] = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"] = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"] = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"] = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"] = tensor[i]
        else:
            out[name] = tensor
    return out

def _rebank_state_dict(sd: dict[str, Tensor], num_layers: int, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert individual 2D tensors back into 3D bank tensors."""
    out: dict[str, Tensor] = {}
    n = num_layers
    qo_slices = [None] * (2 * n)
    kv_slices = [None] * (2 * n)
    up_slices = [None] * n
    down_slices = [None] * n
    consumed = set()
    for i in range(n):
        qk = f"blocks.{i}.attn.c_q.weight"
        if qk in sd:
            qo_slices[i] = sd[qk]
            consumed.add(qk)
        ok = f"blocks.{i}.attn.proj.weight"
        if ok in sd:
            qo_slices[n + i] = sd[ok]
            consumed.add(ok)
        kk = f"blocks.{i}.attn.c_k.weight"
        if kk in sd:
            kv_slices[i] = sd[kk]
            consumed.add(kk)
        vk = f"blocks.{i}.attn.c_v.weight"
        if vk in sd:
            kv_slices[n + i] = sd[vk]
            consumed.add(vk)
        fk = f"blocks.{i}.mlp.fc.weight"
        if fk in sd:
            up_slices[i] = sd[fk]
            consumed.add(fk)
        dk = f"blocks.{i}.mlp.proj.weight"
        if dk in sd:
            down_slices[i] = sd[dk]
            consumed.add(dk)
    out["qo_bank"] = torch.stack(qo_slices).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv_slices).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up_slices).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down_slices).to(dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out

def mixed_quantize_int6(
    state_dict: dict[str, Tensor],
    int6_cats: set[str],
    clip_range: int = 31,
    hessians: dict[str, Tensor] | None = None,
    gptq_block_size: int = 128,
    gptq_percdamp: float = 0.01,
    int8_attn_layers: set[int] | None = None,
    int8_mlp_layers: set[int] | None = None,
    lqer_enabled: bool = False,
    lqer_rank: int = 4,
    lqer_top_k: int = 3,
    lqer_asym_group: int = 32,
):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    gptq_count, naive_count = 0, 0
    lqer_heap: list[tuple[float, int, str, Tensor, Tensor, Tensor]] = []
    lqer_counter = 0
    lqer_all = lqer_top_k <= 0
    int8_attn_layers = int8_attn_layers or set()
    int8_mlp_layers = int8_mlp_layers or set()
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        layer_idx = layer_idx_from_name(name)
        force_int8 = (
            layer_idx is not None
            and ((cat == "attn" and layer_idx in int8_attn_layers) or (cat == "mlp" and layer_idx in int8_mlp_layers))
        )
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1 and not force_int8:
            H = hessians.get(name) if hessians else None
            if H is not None and t.ndim == 2:
                q, s = gptq_quantize_weight(
                    t,
                    H.cpu(),
                    clip_range=clip_range,
                    block_size=gptq_block_size,
                    percdamp=gptq_percdamp,
                )
                gptq_count += 1
            else:
                q, s = quantize_int6_per_row(t, clip_range=clip_range)
                naive_count += 1
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
            if lqer_enabled and t.ndim == 2 and lqer_rank > 0:
                score = _lqer_residual_score(t, q, s)
                item = (score, lqer_counter, name, t, q, s)
                if lqer_all:
                    lqer_heap.append(item)
                elif lqer_top_k > 0 and len(lqer_heap) < lqer_top_k:
                    heapq.heappush(lqer_heap, item)
                elif lqer_top_k > 0 and score > lqer_heap[0][0]:
                    heapq.heapreplace(lqer_heap, item)
                lqer_counter += 1
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    lqer_names: list[str] = []
    if lqer_enabled and lqer_heap:
        selected = sorted(lqer_heap, key=lambda item: item[0], reverse=True)
        for _score, _counter, name, W, q, s in selected:
            pieces = _lqer_factorize_residual(
                W,
                q,
                s,
                rank=lqer_rank,
                asym_group=lqer_asym_group,
            )
            info = meta.get(name)
            if not isinstance(info, dict) or info.get("type") != "int6" or not pieces:
                continue
            if "mean" in pieces:
                result[name + ".lqer_mean"] = pieces["mean"]
                info["lqer_asym_group"] = int(lqer_asym_group)
            if "a" in pieces and "b" in pieces:
                result[name + ".lqer_a"] = pieces["a"]
                result[name + ".lqer_b"] = pieces["b"]
                info["lqer_rank"] = int(pieces["a"].shape[1])
            info["lqer"] = True
            lqer_names.append(name)
    if hessians:
        print(f"gptq_quantize: {gptq_count} GPTQ layers, {naive_count} naive layers", flush=True)
    if lqer_names:
        print(
            f"lqer:corrected {len(lqer_names)} tensors rank={lqer_rank} "
            f"asym_group={lqer_asym_group} names={lqer_names}",
            flush=True,
        )
    if int8_attn_layers or int8_mlp_layers:
        print(
            f"mixed_quantize:int8_attn_layers={sorted(int8_attn_layers)} "
            f"int8_mlp_layers={sorted(int8_mlp_layers)}",
            flush=True,
        )
    return result, meta
def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        deq = _dequant_row_scaled(q, s)
        if isinstance(info, dict) and info.get("lqer"):
            group = int(info.get("lqer_asym_group", 0))
            mean_key = name + ".lqer_mean"
            if group > 0 and mean_key in result and deq.ndim == 2:
                deq = _apply_lqer_group(deq, result[mean_key], group)
            a_key, b_key = name + ".lqer_a", name + ".lqer_b"
            if a_key in result and b_key in result:
                deq = deq + result[a_key].float() @ result[b_key].float()
        out[name] = deq.to(orig_dtype)
    return out

# --- Full Hessian GPTQ ---

def gptq_quantize_weight(W: Tensor, H: Tensor, clip_range: int = 31,
                          block_size: int = 128, percdamp: float = 0.01) -> tuple[Tensor, Tensor]:
    """GPTQ with Cholesky error compensation and actorder (Frantar et al., ICLR 2023)."""
    W_orig = W.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = percdamp * H.diag().mean()
    H.diagonal().add_(damp)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    try:
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except torch.linalg.LinAlgError:
        return quantize_int6_per_row(W_orig, clip_range)
    best_q, best_scale, best_err = None, None, float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(W_orig.abs(), pct, dim=1)
        else:
            row_clip = W_orig.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        Q = torch.zeros(rows, cols, dtype=torch.int8)
        W_work = W_perm.clone()
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            W_block = W_work[:, i1:i2].clone()
            Hinv_block = Hinv[i1:i2, i1:i2]
            Err = torch.zeros(rows, i2 - i1)
            for j in range(i2 - i1):
                w_col = W_block[:, j]
                d = Hinv_block[j, j]
                q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
                Q[:, i1 + j] = q_col.to(torch.int8)
                err = (w_col - q_col.float() * sf) / d
                Err[:, j] = err
                W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
            if i2 < cols:
                W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        mse = (W_perm - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    best_q = best_q[:, invperm]
    return best_q, best_scale

def _init_hessians(nl: int, dim: int, mlp_dim: int, device: torch.device) -> dict[str, Tensor]:
    h: dict[str, Tensor] = {}
    for i in range(nl):
        for k in ['c_q', 'c_k', 'c_v']:
            h[f'blocks.{i}.attn.{k}.weight'] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        h[f'blocks.{i}.attn.proj.weight'] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        h[f'blocks.{i}.mlp.fc.weight'] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        h[f'blocks.{i}.mlp.proj.weight'] = torch.zeros(mlp_dim, mlp_dim, dtype=torch.float32, device=device)
    return h

def _accum_hessians(hessians: dict[str, Tensor], blocks: nn.ModuleList, dim: int, mlp_dim: int) -> None:
    for i, block in enumerate(blocks):
        qkv_in = block.attn._gptq_qkv_in.float().reshape(-1, dim)
        h_qkv = qkv_in.t() @ qkv_in
        hessians[f'blocks.{i}.attn.c_q.weight'] += h_qkv
        hessians[f'blocks.{i}.attn.c_k.weight'] += h_qkv
        hessians[f'blocks.{i}.attn.c_v.weight'] += h_qkv
        o_in = block.attn._gptq_o_in.float().reshape(-1, dim)
        hessians[f'blocks.{i}.attn.proj.weight'] += o_in.t() @ o_in
        up_in = block.mlp._gptq_up_in.float().reshape(-1, dim)
        hessians[f'blocks.{i}.mlp.fc.weight'] += up_in.t() @ up_in
        down_in = block.mlp._gptq_down_in.float().reshape(-1, mlp_dim)
        hessians[f'blocks.{i}.mlp.proj.weight'] += down_in.t() @ down_in

def _finalize_hessians(hessians: dict[str, Tensor], num_batches: int) -> None:
    for name in hessians:
        hessians[name] = hessians[name].cpu() / num_batches
        damp = 0.01 * torch.diag(hessians[name]).mean().clamp_min(1e-6)
        hessians[name] += damp * torch.eye(hessians[name].shape[0])

def gptq_collect_hessians(base_model: nn.Module, train_loader, device: torch.device,
                           num_batches: int, batch_tokens: int, seq_len: int,
                           grad_accum_steps: int) -> dict[str, Tensor]:
    """Collect Hessians H = X^T X from training batches during the counted training phase."""
    nl = base_model.num_layers
    dim = base_model.tok_emb.weight.shape[1]
    mlp_dim = base_model.mlp_up_bank.shape[1]
    hessians = _init_hessians(nl, dim, mlp_dim, device)
    for block in base_model.blocks:
        block.attn._save_gptq = True
        block.mlp._save_gptq = True
    base_model.eval()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for _ in range(num_batches):
            x, y = train_loader.next_batch(batch_tokens, seq_len, grad_accum_steps)
            base_model(x, y)
            _accum_hessians(hessians, base_model.blocks, dim, mlp_dim)
    for block in base_model.blocks:
        block.attn._save_gptq = False
        block.mlp._save_gptq = False
    _finalize_hessians(hessians, num_batches)
    base_model.train()
    return hessians

# --- Online Best-Agree Eval (CTW + multi-expert overlay) ---

WHITESPACE_BYTE_IDS = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 36}
_EDGE_PUNCT = ".,:;!?()[]{}<>\"'`"
_ROLLING_COEFFS = (36313, 27191, 51647, 81929, 131071, 196613, 262147, 393241, 524309, 655373, 786433, 917521, 1048583, 1179653, 1310729, 1441801, 1572869, 1703941, 1835017, 1966087, 2097169, 2228243, 2359319, 2490389, 2621471, 2752549, 2883617, 3014687, 3145757, 3276833, 3407903, 3538973)
_PREFIX_BASE, _LEN_MIX, _PAIR_MIX, _MASK64 = 1099511628211, 0x9E3779B185EBCA87, 1000003, (1 << 64) - 1

def _normalize_word(text: str, mode: str) -> str:
    text = text.strip()
    if mode == "lower": return text.lower()
    if mode == "identity": return text
    if mode == "strip_punct_lower": return text.strip(_EDGE_PUNCT).lower()
    raise ValueError(f"Unknown word normalization mode: {mode}")

def _apply_boost(llm_true, llm_hint, hit_mask, gate_mask, boost):
    boosted = llm_true.astype(np.float64, copy=True)
    if not gate_mask.any(): return boosted
    if np.isscalar(boost):
        s = math.exp(float(boost))
        hg, mg = gate_mask & hit_mask, gate_mask & ~hit_mask
        boosted[hg] = (s * llm_true[hg]) / (1.0 - llm_true[hg] + s * llm_true[hg])
        boosted[mg] = llm_true[mg] / (1.0 - llm_hint[mg] + s * llm_hint[mg])
        return boosted
    ba = boost.astype(np.float64, copy=False)
    scale = np.ones(llm_true.shape, dtype=np.float64)
    scale[gate_mask] = np.exp(ba[gate_mask])
    boosted = llm_true / (1.0 - llm_hint + scale * llm_hint)
    boosted[gate_mask & hit_mask] *= scale[gate_mask & hit_mask]
    return boosted

def _expected_gain(top_prob, llm_hint_prob, boost):
    q = np.clip(llm_hint_prob.astype(np.float64, copy=False), 1e-12, 1.0 - 1e-12)
    p = np.clip(top_prob.astype(np.float64, copy=False), 0.0, 1.0)
    return (p * boost - np.log1p(q * (math.exp(boost) - 1.0))).astype(np.float32)

def _expert_rel_weight(hits, uses, power):
    return float(np.clip((max((hits + 2.0) / (uses + 4.0) / 0.5, 0.25)) ** power, 0.25, 4.0))

def _compute_best_agree(*, llm_chunk, true_targets, experts, agree_add_boost, min_eg, disagree_scale):
    if not experts:
        e = np.zeros(llm_chunk.shape[0], dtype=np.float32)
        return llm_chunk.astype(np.float64, copy=True), e, np.zeros(llm_chunk.shape[0], dtype=np.bool_)
    ct = np.stack([np.asarray(x["top_token"], dtype=np.uint16) for x in experts], axis=0)
    ch = np.stack([np.asarray(x["hint_probs"], dtype=np.float32) for x in experts], axis=0).astype(np.float64, copy=False)
    cs = np.zeros((len(experts), llm_chunk.shape[0]), dtype=np.float32)
    cc = np.zeros_like(cs, dtype=np.uint8)
    cb = np.zeros_like(cs, dtype=np.float32)
    for ei, ex in enumerate(experts):
        g = np.asarray(ex["gate"], dtype=np.bool_)
        eg = _expected_gain(np.asarray(ex["top_prob"], dtype=np.float32), np.asarray(ex["hint_probs"], dtype=np.float32), float(ex["boost"])) * float(ex["weight"])
        sup = g[None, :] & (ct == np.asarray(ex["top_token"], dtype=np.uint16)[None, :])
        cs += np.where(sup, eg[None, :], 0.0)
        cc += sup.astype(np.uint8)
        cb = np.maximum(cb, np.where(sup, float(ex["boost"]), 0.0))
    ci = np.argmax(cs, axis=0)
    cols = np.arange(llm_chunk.shape[0])
    chosen_s, chosen_c = cs[ci, cols], cc[ci, cols]
    chosen_tok = ct[ci, cols]
    chosen_boost = cb[ci, cols].astype(np.float64) + agree_add_boost * np.maximum(chosen_c.astype(np.int16) - 1, 0)
    dp = np.max(np.where(ct != chosen_tok[None, :], cs, 0.0), axis=0)
    chosen_boost = np.maximum(chosen_boost - disagree_scale * dp, 0.0)
    gate = (chosen_c > 0) & (chosen_s > min_eg)
    boosted = _apply_boost(llm_chunk, ch[ci, cols], chosen_tok == true_targets, gate, chosen_boost)
    return boosted, chosen_s, chosen_c > 0

def _dist_max(v, device, ws):
    if ws <= 1: return float(v)
    t = torch.tensor([v], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())

def _build_chunk_windows(total, seq_len, stride, chunk_sz):
    ws_list = [w for w in range(0, total, stride) if min(w + seq_len, total) - w >= stride or w == 0]
    nc = (total + chunk_sz - 1) // chunk_sz
    cw: list[list[int]] = [[] for _ in range(nc)]
    for w in ws_list:
        s = 0 if w == 0 else max(min(w + seq_len, total) - w - stride, 0)
        cw[min((w + s) // chunk_sz, nc - 1)].append(w)
    return cw

class OnlineFullVocabNgramState:
    """Prefix-only full-vocab token cache.

    For each order d, the predictive distribution is
        q_d(a | ctx) = (count(ctx, a) + alpha) / (count(ctx) + alpha * V).
    The exported score uses a convex mixture of the model distribution and a
    q_d mixture whose weights depend only on prefix statistics, never on the
    realized target token. This gives a proper full-vocabulary distribution
    while still letting scoring gather q(target) cheaply.
    """
    def __init__(
        self,
        *,
        orders,
        vocab_size,
        alpha,
        threshold,
        max_lambda,
        weight_power,
        seed_prefix_token,
    ):
        self.depths = sorted({max(int(o) - 1, 0) for o in orders if int(o) > 0})
        self.vocab_size = int(vocab_size)
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.max_lambda = float(max_lambda)
        self.weight_power = float(weight_power)
        self.hist: list[int] = [int(seed_prefix_token)]
        self.ctx_total: list[dict[int, int]] = [{} for _ in self.depths]
        self.pair_count: list[dict[int, int]] = [{} for _ in self.depths]
        self.best_tok: list[dict[int, int]] = [{} for _ in self.depths]
        self.best_count: list[dict[int, int]] = [{} for _ in self.depths]

    def _ctx_hash(self, d):
        n = len(self.hist)
        if n < d:
            return None
        h, b = 0, n - d
        for j in range(d):
            h ^= (self.hist[b + j] + 1) * _ROLLING_COEFFS[j % len(_ROLLING_COEFFS)]
        return h & _MASK64

    def _pair_key(self, ctx_hash, token):
        return (ctx_hash * _PAIR_MIX ^ (int(token) + 1) * 65537) & _MASK64

    def process_chunk(self, chunk_tokens):
        ct = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        n = int(ct.size)
        q_gold = np.full(n, 1.0 / max(self.vocab_size, 1), dtype=np.float32)
        lam = np.zeros(n, dtype=np.float32)
        q_top = np.zeros(n, dtype=np.float32)
        best_hit = np.zeros(n, dtype=np.bool_)
        order_used = np.zeros(n, dtype=np.uint8)
        denom_prior = self.alpha * self.vocab_size
        for i in range(n):
            tok = int(ct[i])
            ctx_hashes = [self._ctx_hash(d) for d in self.depths]
            w_sum = 0.0
            q_sum = 0.0
            strength_sum = 0.0
            local_top = 0.0
            local_hit = False
            local_order = 0
            for di, (d, h) in enumerate(zip(self.depths, ctx_hashes, strict=True)):
                if h is None:
                    continue
                total = self.ctx_total[di].get(h, 0)
                if total <= 0:
                    continue
                denom = total + denom_prior
                pk = self._pair_key(h, tok)
                qy = (self.pair_count[di].get(pk, 0) + self.alpha) / denom
                bt = self.best_tok[di].get(h, 0)
                bc = self.best_count[di].get(h, 0)
                qt = (bc + self.alpha) / denom
                rel = max((qt - self.threshold) / max(1.0 - self.threshold, 1e-6), 0.0)
                if rel <= 0.0:
                    continue
                strength = rel ** self.weight_power
                w_sum += strength
                q_sum += strength * qy
                strength_sum += strength
                if qt > local_top:
                    local_top = qt
                    local_hit = bt == tok
                    local_order = d + 1
            if w_sum > 0.0:
                q_gold[i] = q_sum / w_sum
                lam[i] = min(self.max_lambda, self.max_lambda * min(strength_sum, 1.0))
                q_top[i] = local_top
                best_hit[i] = local_hit
                order_used[i] = min(local_order, 255)
            for di, h in enumerate(ctx_hashes):
                if h is None:
                    continue
                pk = self._pair_key(h, tok)
                pc = self.pair_count[di].get(pk, 0) + 1
                self.pair_count[di][pk] = pc
                total = self.ctx_total[di].get(h, 0) + 1
                self.ctx_total[di][h] = total
                if pc > self.best_count[di].get(h, 0):
                    self.best_count[di][h] = pc
                    self.best_tok[di][h] = tok
            self.hist.append(tok)
        return q_gold, lam, q_top, best_hit, order_used

class OnlineRecencyUnigramState:
    """Prefix-only recent-token unigram expert.

    This is a real full-vocabulary distribution:
        q(a) = (recent_count(a) + alpha) / (window_tokens + alpha * V).
    The gate uses only the most frequent token probability in the prefix window.
    """
    def __init__(
        self,
        *,
        vocab_size,
        alpha,
        threshold,
        max_lambda,
        weight_power,
        window,
        seed_prefix_token,
    ):
        self.vocab_size = int(vocab_size)
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.max_lambda = float(max_lambda)
        self.weight_power = float(weight_power)
        self.window = max(int(window), 0)
        self.queue: deque[int] = deque()
        self.counts: dict[int, int] = {}
        self.best_token = 0
        self.best_count = 0
        if self.window > 0:
            self._push(int(seed_prefix_token))

    def _refresh_best(self) -> None:
        if not self.counts:
            self.best_token = 0
            self.best_count = 0
            return
        self.best_token, self.best_count = max(self.counts.items(), key=lambda kv: kv[1])

    def _push(self, tok: int) -> None:
        if self.window <= 0:
            return
        tok = int(tok)
        self.queue.append(tok)
        c = self.counts.get(tok, 0) + 1
        self.counts[tok] = c
        if c > self.best_count:
            self.best_token = tok
            self.best_count = c
        while len(self.queue) > self.window:
            old = self.queue.popleft()
            oc = self.counts.get(old, 0) - 1
            if oc <= 0:
                self.counts.pop(old, None)
            else:
                self.counts[old] = oc
            if old == self.best_token:
                self._refresh_best()

    def process_chunk(self, chunk_tokens):
        ct = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        n = int(ct.size)
        q_gold = np.full(n, 1.0 / max(self.vocab_size, 1), dtype=np.float32)
        lam = np.zeros(n, dtype=np.float32)
        q_top = np.zeros(n, dtype=np.float32)
        best_hit = np.zeros(n, dtype=np.bool_)
        order_used = np.ones(n, dtype=np.uint8)
        denom_prior = self.alpha * self.vocab_size
        for i in range(n):
            tok = int(ct[i])
            total = len(self.queue)
            if total > 0:
                denom = total + denom_prior
                qy = (self.counts.get(tok, 0) + self.alpha) / denom
                qt = (self.best_count + self.alpha) / denom
                rel = max((qt - self.threshold) / max(1.0 - self.threshold, 1e-6), 0.0)
                if rel > 0.0:
                    q_gold[i] = qy
                    lam[i] = min(self.max_lambda, self.max_lambda * (rel ** self.weight_power))
                    q_top[i] = qt
                    best_hit[i] = self.best_token == tok
            self._push(tok)
        return q_gold, lam, q_top, best_hit, order_used

class OnlineCausalCompressionArbiterState:
    """Prefix-grown expert arbiter.

    Each expert contributes a normalized full-vocabulary distribution and a
    prefix-only weight. The arbiter caps the total expert mass and returns a
    single normalized q_mix so the caller can keep using:
        p_final = (1 - lambda) * p_model + lambda * q_mix.
    """
    def __init__(
        self,
        *,
        ngram: OnlineFullVocabNgramState | None,
        recency: OnlineRecencyUnigramState | None,
        max_lambda,
        ngram_scale,
        recency_scale,
    ):
        self.ngram = ngram
        self.recency = recency
        self.max_lambda = float(max_lambda)
        self.ngram_scale = float(ngram_scale)
        self.recency_scale = float(recency_scale)
        self.vocab_size = (
            ngram.vocab_size if ngram is not None
            else recency.vocab_size if recency is not None
            else 1
        )

    def describe(self) -> str:
        parts = ["arbiter"]
        if self.ngram is not None:
            parts.append(f"ngramx{self.ngram_scale:g}")
        if self.recency is not None:
            parts.append(f"recency{self.recency.window}x{self.recency_scale:g}")
        parts.append(f"cap{self.max_lambda:g}")
        return "+".join(parts)

    def _add_expert(self, q_acc, lam_acc, q_top_acc, hit_acc, order_acc, q, lam, q_top, hit, order, scale):
        w = np.clip(np.asarray(lam, dtype=np.float32) * float(scale), 0.0, 1.0)
        q_acc += w.astype(np.float64) * np.asarray(q, dtype=np.float64)
        lam_acc += w.astype(np.float64)
        better = np.asarray(q_top, dtype=np.float32) > q_top_acc
        q_top_acc[better] = np.asarray(q_top, dtype=np.float32)[better]
        hit_acc[better] = np.asarray(hit, dtype=np.bool_)[better]
        order_acc[better] = np.asarray(order, dtype=np.uint8)[better]

    def process_chunk(self, chunk_tokens):
        n = int(chunk_tokens.size)
        q_acc = np.zeros(n, dtype=np.float64)
        lam_acc = np.zeros(n, dtype=np.float64)
        q_top = np.zeros(n, dtype=np.float32)
        hit = np.zeros(n, dtype=np.bool_)
        order = np.zeros(n, dtype=np.uint8)
        if self.ngram is not None:
            q, lam, qt, ht, od = self.ngram.process_chunk(chunk_tokens)
            self._add_expert(q_acc, lam_acc, q_top, hit, order, q, lam, qt, ht, od, self.ngram_scale)
        if self.recency is not None:
            q, lam, qt, ht, od = self.recency.process_chunk(chunk_tokens)
            self._add_expert(q_acc, lam_acc, q_top, hit, order, q, lam, qt, ht, od, self.recency_scale)
        over = lam_acc > self.max_lambda
        if over.any():
            shrink = self.max_lambda / np.maximum(lam_acc[over], 1e-12)
            q_acc[over] *= shrink
            lam_acc[over] = self.max_lambda
        q_mix = np.full(n, 1.0 / max(self.vocab_size, 1), dtype=np.float32)
        used = lam_acc > 0.0
        q_mix[used] = (q_acc[used] / lam_acc[used]).astype(np.float32)
        return q_mix, lam_acc.astype(np.float32), q_top, hit, order

_NATIVE_FULL_VOCAB_CACHE_SRC = Path(__file__).with_name("online_full_vocab_cache.c")
_NATIVE_FULL_VOCAB_CACHE_LIBS: dict[int, ctypes.CDLL] = {}

def _ensure_native_full_vocab_cache_lib(rank: int, log0=print) -> ctypes.CDLL:
    cached = _NATIVE_FULL_VOCAB_CACHE_LIBS.get(rank)
    if cached is not None:
        return cached
    if not _NATIVE_FULL_VOCAB_CACHE_SRC.exists():
        raise FileNotFoundError(str(_NATIVE_FULL_VOCAB_CACHE_SRC))
    default_lib_dir = "/tmp" if os.name != "nt" else str(_NATIVE_FULL_VOCAB_CACHE_SRC.parent)
    lib_dir = Path(os.environ.get("NATIVE_FULL_VOCAB_LIB_DIR", default_lib_dir))
    lib_dir.mkdir(parents=True, exist_ok=True)
    src_mtime = _NATIVE_FULL_VOCAB_CACHE_SRC.stat().st_mtime_ns
    lib_path = lib_dir / f"parameter_golf_full_vocab_cache_{src_mtime}_r{rank}.so"
    needs_build = not lib_path.exists()
    if needs_build:
        log0(f"native_full_vocab_cache:building src={_NATIVE_FULL_VOCAB_CACHE_SRC.name} out={lib_path.name}")
        subprocess.run(
            [
                "gcc", "-O3", "-march=native", "-std=c11", "-shared", "-fPIC",
                "-o", str(lib_path), str(_NATIVE_FULL_VOCAB_CACHE_SRC), "-lm",
            ],
            check=True,
        )
    lib = ctypes.CDLL(str(lib_path))
    lib.fv_cache_create.restype = ctypes.c_void_p
    lib.fv_cache_create.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_uint16,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.fv_cache_destroy.restype = None
    lib.fv_cache_destroy.argtypes = [ctypes.c_void_p]
    lib.fv_cache_process_chunk.restype = ctypes.c_int
    lib.fv_cache_process_chunk.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    _NATIVE_FULL_VOCAB_CACHE_LIBS[rank] = lib
    return lib

class NativeOnlinePrefixCache:
    def __init__(
        self,
        *,
        orders,
        vocab_size,
        alpha,
        threshold,
        max_lambda,
        weight_power,
        seed_prefix_token,
        use_ngram,
        use_recency,
        arbiter_max_lambda,
        ngram_scale,
        recency_scale,
        recency_alpha,
        recency_threshold,
        recency_max_lambda,
        recency_weight_power,
        recency_window,
        table_bits,
        rank,
        log0=print,
    ):
        self.lib = _ensure_native_full_vocab_cache_lib(rank, log0=log0)
        clean_orders = sorted({int(o) for o in orders if int(o) > 0})
        if not clean_orders and use_ngram:
            clean_orders = [1]
        order_arr = np.ascontiguousarray(clean_orders, dtype=np.int32)
        self.state = self.lib.fv_cache_create(
            order_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(int(order_arr.size)),
            ctypes.c_int(int(vocab_size)),
            ctypes.c_double(float(alpha)),
            ctypes.c_double(float(threshold)),
            ctypes.c_double(float(max_lambda)),
            ctypes.c_double(float(weight_power)),
            ctypes.c_uint16(int(seed_prefix_token) & 0xFFFF),
            ctypes.c_int(int(bool(use_ngram))),
            ctypes.c_int(int(bool(use_recency))),
            ctypes.c_double(float(arbiter_max_lambda)),
            ctypes.c_double(float(ngram_scale)),
            ctypes.c_double(float(recency_scale)),
            ctypes.c_double(float(recency_alpha)),
            ctypes.c_double(float(recency_threshold)),
            ctypes.c_double(float(recency_max_lambda)),
            ctypes.c_double(float(recency_weight_power)),
            ctypes.c_int(int(recency_window)),
            ctypes.c_int(int(table_bits)),
        )
        if not self.state:
            raise RuntimeError(
                f"native full-vocab cache allocation failed table_bits={table_bits} "
                f"orders={clean_orders} recency_window={recency_window}"
            )
        self.orders = clean_orders
        self.vocab_size = int(vocab_size)
        self.use_ngram = bool(use_ngram)
        self.use_recency = bool(use_recency)
        self.table_bits = int(table_bits)
        self.recency_window = int(recency_window)

    def close(self) -> None:
        if getattr(self, "state", None):
            self.lib.fv_cache_destroy(self.state)
            self.state = None

    def __del__(self):
        self.close()

    def describe(self) -> str:
        parts = [f"native_c_bits{self.table_bits}"]
        if self.use_ngram:
            parts.append("ngram" + ",".join(str(o) for o in self.orders))
        if self.use_recency:
            parts.append(f"recency{self.recency_window}")
        return "+".join(parts)

    def process_chunk(self, chunk_tokens):
        ct = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        n = int(ct.size)
        q_mix = np.empty(n, dtype=np.float32)
        lam = np.empty(n, dtype=np.float32)
        q_top = np.empty(n, dtype=np.float32)
        hit = np.empty(n, dtype=np.uint8)
        order = np.empty(n, dtype=np.uint8)
        rc = self.lib.fv_cache_process_chunk(
            self.state,
            ct.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int64(n),
            q_mix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            lam.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            q_top.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            hit.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            order.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if rc != 0:
            raise RuntimeError(f"native full-vocab cache process_chunk failed rc={rc}")
        return q_mix, lam, q_top, hit.astype(np.bool_), order

def _make_online_prefix_cache(args, tok_np, orders, alpha, threshold, max_lam, weight_power,
                              log0=print, rank: int = 0):
    E = os.environ.get
    use_arbiter = bool(int(E("CAUSAL_ARBITER_EVAL", "1")))
    native_enabled = bool(int(E("NATIVE_FULL_VOCAB_CACHE", "1")))
    use_ngram = (not use_arbiter) or bool(int(E("ARBITER_NGRAM_ENABLED", "1")))
    use_recency = use_arbiter and bool(int(E("ARBITER_RECENCY_ENABLED", "1")))
    rec_alpha = float(E("RECENCY_CACHE_ALPHA", E("FULL_VOCAB_CACHE_ALPHA", "0.05")))
    rec_threshold = float(E("RECENCY_CACHE_THRESHOLD", "0.08"))
    rec_max_lambda = float(E("RECENCY_CACHE_MAX_LAMBDA", "0.12"))
    rec_weight_power = float(E("RECENCY_CACHE_WEIGHT_POWER", "1.25"))
    rec_window = int(E("RECENCY_CACHE_WINDOW", "4096"))
    arbiter_max_lambda = float(E("ARBITER_MAX_LAMBDA", str(max_lam))) if use_arbiter else max_lam
    ngram_scale = float(E("ARBITER_NGRAM_SCALE", "1.0")) if use_arbiter else 1.0
    recency_scale = float(E("ARBITER_RECENCY_SCALE", "1.0")) if use_arbiter else 0.0
    if native_enabled:
        try:
            return NativeOnlinePrefixCache(
                orders=orders,
                vocab_size=args.vocab_size,
                alpha=alpha,
                threshold=threshold,
                max_lambda=max_lam,
                weight_power=weight_power,
                seed_prefix_token=int(tok_np[0]),
                use_ngram=use_ngram,
                use_recency=use_recency,
                arbiter_max_lambda=arbiter_max_lambda,
                ngram_scale=ngram_scale,
                recency_scale=recency_scale,
                recency_alpha=rec_alpha,
                recency_threshold=rec_threshold,
                recency_max_lambda=rec_max_lambda,
                recency_weight_power=rec_weight_power,
                recency_window=rec_window,
                table_bits=int(E("NATIVE_FULL_VOCAB_TABLE_BITS", "26")),
                rank=rank,
                log0=log0,
            )
        except Exception as exc:
            log0(f"native_full_vocab_cache:fallback_to_python reason={type(exc).__name__}:{exc}")
    if not use_arbiter:
        return OnlineFullVocabNgramState(
            orders=orders,
            vocab_size=args.vocab_size,
            alpha=alpha,
            threshold=threshold,
            max_lambda=max_lam,
            weight_power=weight_power,
            seed_prefix_token=int(tok_np[0]),
        )
    ngram = None
    recency = None
    if use_ngram:
        ngram = OnlineFullVocabNgramState(
            orders=orders,
            vocab_size=args.vocab_size,
            alpha=alpha,
            threshold=threshold,
            max_lambda=max_lam,
            weight_power=weight_power,
            seed_prefix_token=int(tok_np[0]),
        )
    if use_recency:
        recency = OnlineRecencyUnigramState(
            vocab_size=args.vocab_size,
            alpha=rec_alpha,
            threshold=rec_threshold,
            max_lambda=rec_max_lambda,
            weight_power=rec_weight_power,
            window=rec_window,
            seed_prefix_token=int(tok_np[0]),
        )
    return OnlineCausalCompressionArbiterState(
        ngram=ngram,
        recency=recency,
        max_lambda=arbiter_max_lambda,
        ngram_scale=ngram_scale,
        recency_scale=recency_scale,
    )

def _modulate_lambda_by_model_confidence(lam, model_top, power, floor):
    if power <= 0.0:
        return lam
    mt = np.clip(np.asarray(model_top, dtype=np.float64), 0.0, 1.0)
    floor = float(np.clip(floor, 0.0, 0.99))
    rel = np.clip((mt - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0)
    return np.asarray(lam, dtype=np.float64) * ((1.0 - rel) ** float(power))

def eval_val_sliding_online_full_vocab_mix(
    *, args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride, batch_seqs=32, eval_seq_len=None, log0=print,
):
    t0 = time.perf_counter()
    seq_len = eval_seq_len or args.train_seq_len
    E = os.environ.get
    chunk_sz = int(E("FULL_VOCAB_CACHE_CHUNK_TOKENS", E("CHUNK_TOKENS", "131072")))
    orders = tuple(int(x) for x in E("FULL_VOCAB_CACHE_ORDERS", "4,8,16,32").split(",") if x.strip())
    alpha = float(E("FULL_VOCAB_CACHE_ALPHA", "0.05"))
    threshold = float(E("FULL_VOCAB_CACHE_THRESHOLD", "0.28"))
    max_lam = float(E("FULL_VOCAB_CACHE_MAX_LAMBDA", "0.35"))
    weight_power = float(E("FULL_VOCAB_CACHE_WEIGHT_POWER", "1.5"))
    model_conf_power = float(E("ARBITER_MODEL_CONFIDENCE_POWER", "0.5"))
    model_conf_floor = float(E("ARBITER_MODEL_CONFIDENCE_FLOOR", "0.10"))
    total_tgt = val_tokens.numel() - 1
    tok_np = val_tokens.cpu().numpy().astype(np.uint16, copy=False)
    _log = log0 if rank == 0 else (lambda *_: None)
    cache = _make_online_prefix_cache(
        args, tok_np, orders, alpha, threshold, max_lam, weight_power,
        log0=_log, rank=rank,
    )
    cache_desc = cache.describe() if hasattr(cache, "describe") else "ngram_only"
    compiled = _compile_logits_fn(base_model, seq_len=seq_len, device=device, log0=_log)
    startup_s = _dist_max(time.perf_counter() - t0, device, world_size)
    if rank == 0:
        log0(
            "full_vocab_cache_mix:start "
            f"total_targets={total_tgt} seq_len={seq_len} stride={stride} "
            f"chunk_tokens={chunk_sz} orders={orders} alpha={alpha} "
            f"threshold={threshold} max_lambda={max_lam} cache={cache_desc} "
            f"model_conf_power={model_conf_power} startup_max={startup_s:.2f}s"
        )
    cws = _build_chunk_windows(total_tgt, seq_len, stride, chunk_sz)
    llm_loss_sum = mix_loss_sum = byte_sum = token_count = 0.0
    cache_uses = cache_hits = lambda_sum = top_sum = 0.0
    state_t = input_t = forward_t = blend_t = 0.0
    loop_t0 = time.perf_counter()
    with torch.inference_mode():
        for ci, windows in enumerate(cws):
            if not windows:
                continue
            ct0 = ci * chunk_sz
            ct1 = min((ci + 1) * chunk_sz, total_tgt)
            ctgt = np.ascontiguousarray(tok_np[ct0 + 1 : ct1 + 1], dtype=np.uint16)
            ts0 = time.perf_counter()
            q_gold, lam, q_top, hit, _order = cache.process_chunk(ctgt)
            state_t += time.perf_counter() - ts0
            s_e = ((len(windows) * rank) // world_size, (len(windows) * (rank + 1)) // world_size)
            my_wins = windows[s_e[0] : s_e[1]]
            for bi in range(0, len(my_wins), batch_seqs):
                bws = my_wins[bi : bi + batch_seqs]
                if not bws:
                    continue
                ti0 = time.perf_counter()
                bsz = len(bws)
                x_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens, sstarts = [], []
                for i, ws in enumerate(bws):
                    end = min(ws + seq_len, total_tgt)
                    wlen = end - ws
                    wlens.append(wlen)
                    loc = val_tokens[ws : end + 1].to(device=device, dtype=torch.int64)
                    x_b[i, :wlen] = loc[:-1]
                    y_b[i, :wlen] = loc[1:]
                    sstarts.append(0 if ws == 0 else max(wlen - stride, 0))
                input_t += time.perf_counter() - ti0
                tf0 = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                        torch.compiler.cudagraph_mark_step_begin()
                    logits = compiled(x_b)
                lf = logits.float()
                ln = torch.logsumexp(lf, dim=-1)
                model_p_gold = (lf.gather(-1, y_b.unsqueeze(-1)).squeeze(-1) - ln).exp()
                model_top_prob = (lf.amax(dim=-1) - ln).exp() if model_conf_power > 0.0 else None
                forward_t += time.perf_counter() - tf0
                tb0 = time.perf_counter()
                for i, ws in enumerate(bws):
                    wlen, s = wlens[i], sstarts[i]
                    if wlen - s <= 0:
                        continue
                    c0, c1 = ws + s - ct0, ws + wlen - ct0
                    pm = model_p_gold[i, s:wlen].detach().cpu().numpy().astype(np.float64, copy=False)
                    qy = np.asarray(q_gold[c0:c1], dtype=np.float64)
                    lm = np.asarray(lam[c0:c1], dtype=np.float64)
                    if model_top_prob is not None:
                        mt = model_top_prob[i, s:wlen].detach().cpu().numpy()
                        lm = _modulate_lambda_by_model_confidence(lm, mt, model_conf_power, model_conf_floor)
                    pf = np.clip((1.0 - lm) * pm + lm * qy, 1e-12, 1.0)
                    used = lm > 0
                    cache_uses += float(used.sum())
                    cache_hits += float((np.asarray(hit[c0:c1], dtype=np.bool_) & used).sum())
                    lambda_sum += float(lm.sum())
                    top_sum += float(np.asarray(q_top[c0:c1], dtype=np.float64)[used].sum()) if used.any() else 0.0
                    llm_loss_sum += float((-np.log(np.clip(pm, 1e-12, 1.0))).sum())
                    mix_loss_sum += float((-np.log(pf)).sum())
                    token_count += float(c1 - c0)
                    tgt, prev = y_b[i, s:wlen], x_b[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_sum += float(tb.sum().item())
                blend_t += time.perf_counter() - tb0
    def _f64(v): return torch.tensor([v], dtype=torch.float64, device=device)
    ll_t, mx_t, by_t, tc_t = _f64(llm_loss_sum), _f64(mix_loss_sum), _f64(byte_sum), _f64(token_count)
    use_t, hit_t, lam_t, top_t = _f64(cache_uses), _f64(cache_hits), _f64(lambda_sum), _f64(top_sum)
    if world_size > 1:
        for t in [ll_t, mx_t, by_t, tc_t, use_t, hit_t, lam_t, top_t]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    llm_total, mix_total = float(ll_t.item()), float(mx_t.item())
    bytes_total, tokens_total = float(by_t.item()), float(tc_t.item())
    llm_bpb = llm_total / (bytes_total * math.log(2.0))
    mix_bpb = mix_total / (bytes_total * math.log(2.0))
    uses = float(use_t.item())
    timings = {
        "llm_bpb": llm_bpb,
        "mix_bpb": mix_bpb,
        "gain_bpb": llm_bpb - mix_bpb,
        "llm_nats_per_byte": llm_total / bytes_total,
        "mix_nats_per_byte": mix_total / bytes_total,
        "gain_nats_per_byte": (llm_total - mix_total) / bytes_total,
        "startup_max_s": startup_s,
        "loop_total_max_s": _dist_max(time.perf_counter() - loop_t0, device, world_size),
        "state_max_s": _dist_max(state_t, device, world_size),
        "input_max_s": _dist_max(input_t, device, world_size),
        "forward_max_s": _dist_max(forward_t, device, world_size),
        "blend_max_s": _dist_max(blend_t, device, world_size),
        "cache_use_rate": uses / max(tokens_total, 1.0),
        "cache_hit_rate": float(hit_t.item()) / max(uses, 1.0),
        "avg_lambda": float(lam_t.item()) / max(tokens_total, 1.0),
        "avg_top_prob_when_used": float(top_t.item()) / max(uses, 1.0),
    }
    if rank == 0:
        log0(
            "full_vocab_cache_mix:done "
            f"llm_bpb={llm_bpb:.8f} mix_bpb={mix_bpb:.8f} "
            f"gain_bpb={llm_bpb - mix_bpb:.8f} "
            f"use_rate={timings['cache_use_rate']:.4f} hit_rate={timings['cache_hit_rate']:.4f} "
            f"avg_lambda={timings['avg_lambda']:.4f}"
        )
    return mix_total / max(tokens_total, 1.0), mix_bpb, timings

def eval_val_sliding_prefix_mix_ttt(
    *, args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride, batch_seqs=32, eval_seq_len=None, log0=print,
):
    """Legal combined Track-B evaluator.

    For each validation chunk:
      1. Define prefix-cache probabilities for positions in the chunk from the
         strict prefix stream.
      2. Score the chunk once with the current model and full-vocab cache mix.
      3. Only after scoring, update model parameters on that scored chunk.

    This avoids the invalid pattern of running cache scoring after a full TTT
    pass has already adapted on future validation tokens.
    """
    t0 = time.perf_counter()
    seq_len = eval_seq_len or args.train_seq_len
    prime_rotary_caches(base_model, device, seq_len)
    E = os.environ.get
    chunk_sz = int(E("PREFIX_MIX_TTT_CHUNK_TOKENS", E("TTT_CHUNK_TOKENS", str(args.ttt_chunk_tokens))))
    orders = tuple(int(x) for x in E("FULL_VOCAB_CACHE_ORDERS", "4,8,16,32").split(",") if x.strip())
    alpha = float(E("FULL_VOCAB_CACHE_ALPHA", "0.05"))
    threshold = float(E("FULL_VOCAB_CACHE_THRESHOLD", "0.28"))
    max_lam = float(E("FULL_VOCAB_CACHE_MAX_LAMBDA", "0.35"))
    weight_power = float(E("FULL_VOCAB_CACHE_WEIGHT_POWER", "1.5"))
    model_conf_power = float(E("ARBITER_MODEL_CONFIDENCE_POWER", "0.5"))
    model_conf_floor = float(E("ARBITER_MODEL_CONFIDENCE_FLOOR", "0.10"))
    total_tgt = val_tokens.numel() - 1
    tok_np = val_tokens.cpu().numpy().astype(np.uint16, copy=False)
    _log = log0 if rank == 0 else (lambda *_: None)
    cache = _make_online_prefix_cache(
        args, tok_np, orders, alpha, threshold, max_lam, weight_power,
        log0=_log, rank=rank,
    )
    cache_desc = cache.describe() if hasattr(cache, "describe") else "ngram_only"
    frozen = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        if any(f"blocks.{bi}." in name for bi in frozen):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    compiled_logits = _compile_logits_fn(base_model, seq_len=seq_len, device=device, log0=log0 if rank == 0 else (lambda *_: None))
    startup_s = _dist_max(time.perf_counter() - t0, device, world_size)
    cws = _build_chunk_windows(total_tgt, seq_len, stride, chunk_sz)
    nc = len(cws)
    if rank == 0:
        log0(
            "prefix_mix_ttt:start "
            f"chunks={nc} chunk_tokens={chunk_sz} seq_len={seq_len} stride={stride} "
            f"orders={orders} lambda_max={max_lam} cache={cache_desc} "
            f"model_conf_power={model_conf_power} ttt_lr={args.ttt_lr} "
            f"ttt_epochs={args.ttt_epochs} startup_max={startup_s:.2f}s"
        )
    llm_loss_sum = mix_loss_sum = byte_sum = token_count = 0.0
    cache_uses = cache_hits = lambda_sum = top_sum = 0.0
    state_t = input_t = forward_t = blend_t = train_t = 0.0
    loop_t0 = time.perf_counter()
    for ci, windows in enumerate(cws):
        ct0 = ci * chunk_sz
        ct1 = min((ci + 1) * chunk_sz, total_tgt)
        if ct0 >= ct1:
            continue
        ctgt = np.ascontiguousarray(tok_np[ct0 + 1 : ct1 + 1], dtype=np.uint16)
        ts0 = time.perf_counter()
        q_gold, lam, q_top, hit, _order = cache.process_chunk(ctgt)
        state_t += time.perf_counter() - ts0
        if windows:
            s_e = ((len(windows) * rank) // world_size, (len(windows) * (rank + 1)) // world_size)
            my_wins = windows[s_e[0] : s_e[1]]
            base_model.eval()
            with torch.inference_mode():
                for bi in range(0, len(my_wins), batch_seqs):
                    bws = my_wins[bi : bi + batch_seqs]
                    if not bws:
                        continue
                    ti0 = time.perf_counter()
                    bsz = len(bws)
                    x_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                    y_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                    wlens, sstarts = [], []
                    for i, ws in enumerate(bws):
                        end = min(ws + seq_len, total_tgt)
                        wlen = end - ws
                        wlens.append(wlen)
                        loc = val_tokens[ws : end + 1].to(device=device, dtype=torch.int64)
                        x_b[i, :wlen] = loc[:-1]
                        y_b[i, :wlen] = loc[1:]
                        sstarts.append(0 if ws == 0 else max(wlen - stride, 0))
                    input_t += time.perf_counter() - ti0
                    tf0 = time.perf_counter()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        cudagraph_step_begin()
                        logits = compiled_logits(x_b)
                    lf = logits.float()
                    ln = torch.logsumexp(lf, dim=-1)
                    model_p_gold = (lf.gather(-1, y_b.unsqueeze(-1)).squeeze(-1) - ln).exp()
                    model_top_prob = (lf.amax(dim=-1) - ln).exp() if model_conf_power > 0.0 else None
                    forward_t += time.perf_counter() - tf0
                    tb0 = time.perf_counter()
                    for i, ws in enumerate(bws):
                        wlen, s = wlens[i], sstarts[i]
                        if wlen - s <= 0:
                            continue
                        c0, c1 = ws + s - ct0, ws + wlen - ct0
                        pm = model_p_gold[i, s:wlen].detach().cpu().numpy().astype(np.float64, copy=False)
                        qy = np.asarray(q_gold[c0:c1], dtype=np.float64)
                        lm = np.asarray(lam[c0:c1], dtype=np.float64)
                        if model_top_prob is not None:
                            mt = model_top_prob[i, s:wlen].detach().cpu().numpy()
                            lm = _modulate_lambda_by_model_confidence(lm, mt, model_conf_power, model_conf_floor)
                        pf = np.clip((1.0 - lm) * pm + lm * qy, 1e-12, 1.0)
                        used = lm > 0
                        cache_uses += float(used.sum())
                        cache_hits += float((np.asarray(hit[c0:c1], dtype=np.bool_) & used).sum())
                        lambda_sum += float(lm.sum())
                        top_sum += float(np.asarray(q_top[c0:c1], dtype=np.float64)[used].sum()) if used.any() else 0.0
                        llm_loss_sum += float((-np.log(np.clip(pm, 1e-12, 1.0))).sum())
                        mix_loss_sum += float((-np.log(pf)).sum())
                        token_count += float(c1 - c0)
                        tgt, prev = y_b[i, s:wlen], x_b[i, s:wlen]
                        tb = base_bytes_lut[tgt].to(torch.float64)
                        tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                        byte_sum += float(tb.sum().item())
                    blend_t += time.perf_counter() - tb0
        if ci < nc - 1 and args.ttt_epochs > 0:
            tr0 = time.perf_counter()
            base_model.train()
            chunk_seqs = (ct1 - ct0) // seq_len
            if chunk_seqs > 0:
                lr_now = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(nc - 1, 1)))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                ms2 = (chunk_seqs * rank) // world_size
                me2 = (chunk_seqs * (rank + 1)) // world_size
                for _ep in range(args.ttt_epochs):
                    for bs in range(0, me2 - ms2, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, me2 - ms2)
                        st = ct0 + (ms2 + bs) * seq_len
                        et = ct0 + (ms2 + be) * seq_len + 1
                        if et > val_tokens.numel():
                            continue
                        loc = val_tokens[st:et].to(device=device, dtype=torch.int64)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(loc[:-1].reshape(-1, seq_len), loc[1:].reshape(-1, seq_len))
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()
            train_t += time.perf_counter() - tr0
        if rank == 0 and (ci % 10 == 0 or ci == nc - 1):
            rb = mix_loss_sum / max(byte_sum * math.log(2.0), 1.0)
            log0(f"  prefix_mix_ttt_chunk [{ci+1}/{nc}] mix_bpb={rb:.6f}")
    def _f64(v): return torch.tensor([v], dtype=torch.float64, device=device)
    ll_t, mx_t, by_t, tc_t = _f64(llm_loss_sum), _f64(mix_loss_sum), _f64(byte_sum), _f64(token_count)
    use_t, hit_t, lam_t, top_t = _f64(cache_uses), _f64(cache_hits), _f64(lambda_sum), _f64(top_sum)
    if world_size > 1:
        for t in [ll_t, mx_t, by_t, tc_t, use_t, hit_t, lam_t, top_t]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    llm_total, mix_total = float(ll_t.item()), float(mx_t.item())
    bytes_total, tokens_total = float(by_t.item()), float(tc_t.item())
    llm_bpb = llm_total / (bytes_total * math.log(2.0))
    mix_bpb = mix_total / (bytes_total * math.log(2.0))
    uses = float(use_t.item())
    timings = {
        "llm_bpb": llm_bpb,
        "mix_bpb": mix_bpb,
        "gain_bpb": llm_bpb - mix_bpb,
        "llm_nats_per_byte": llm_total / bytes_total,
        "mix_nats_per_byte": mix_total / bytes_total,
        "gain_nats_per_byte": (llm_total - mix_total) / bytes_total,
        "startup_max_s": startup_s,
        "loop_total_max_s": _dist_max(time.perf_counter() - loop_t0, device, world_size),
        "state_max_s": _dist_max(state_t, device, world_size),
        "input_max_s": _dist_max(input_t, device, world_size),
        "forward_max_s": _dist_max(forward_t, device, world_size),
        "blend_max_s": _dist_max(blend_t, device, world_size),
        "train_max_s": _dist_max(train_t, device, world_size),
        "cache_use_rate": uses / max(tokens_total, 1.0),
        "cache_hit_rate": float(hit_t.item()) / max(uses, 1.0),
        "avg_lambda": float(lam_t.item()) / max(tokens_total, 1.0),
        "avg_top_prob_when_used": float(top_t.item()) / max(uses, 1.0),
    }
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    if rank == 0:
        log0(
            "prefix_mix_ttt:done "
            f"llm_bpb={llm_bpb:.8f} mix_bpb={mix_bpb:.8f} "
            f"gain_bpb={llm_bpb - mix_bpb:.8f} "
            f"use_rate={timings['cache_use_rate']:.4f} hit_rate={timings['cache_hit_rate']:.4f} "
            f"avg_lambda={timings['avg_lambda']:.4f}"
        )
    return mix_total / max(tokens_total, 1.0), mix_bpb, timings

class OnlineNgramState:
    def __init__(self, *, token_ctx_len, starts_new_word_lut, boundary_lut, seed_prefix_token):
        self.ctx_len = max(token_ctx_len, 0)
        self.snw, self.bnd = starts_new_word_lut, boundary_lut
        self.ring = [0] * max(self.ctx_len, 1)
        self.head = self.prefix_len = 0
        self.tc_total: dict[int, int] = {}
        self.tc_pair: dict[int, int] = {}
        self.tc_best_t: dict[int, int] = {}
        self.tc_best_c: dict[int, int] = {}
        self.wh = self.wl = 0
        self.wc_total: dict[int, int] = {}
        self.wc_pair: dict[int, int] = {}
        self.wc_best_t: dict[int, int] = {}
        self.wc_best_c: dict[int, int] = {}
        self._push(seed_prefix_token)
    def close(self): pass
    def _ctx_hash(self):
        h = 0
        for j in range(self.ctx_len):
            h ^= self.ring[(self.head + j) % self.ctx_len] * _ROLLING_COEFFS[j % 32]
        return h & _MASK64
    def _push(self, tok):
        if self.ctx_len <= 0: return
        if self.prefix_len < self.ctx_len:
            self.ring[self.prefix_len] = tok; self.prefix_len += 1
        else:
            self.ring[self.head] = tok; self.head = (self.head + 1) % self.ctx_len
    def process_chunk(self, chunk_tokens):
        ct = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        n = int(ct.size)
        tt, tp = np.zeros(n, dtype=np.uint16), np.zeros(n, dtype=np.float32)
        wt, wp, wv = np.zeros(n, dtype=np.uint16), np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.bool_)
        for i in range(n):
            tok = int(ct[i])
            ib, inw = bool(self.bnd[tok]), bool(self.snw[tok])
            if self.ctx_len == 0 or self.prefix_len >= self.ctx_len:
                ck = self._ctx_hash()
                tot = self.tc_total.get(ck, 0)
                if tot > 0: tt[i], tp[i] = self.tc_best_t[ck], self.tc_best_c[ck] / tot
                pk = (ck * _PAIR_MIX ^ tok * _ROLLING_COEFFS[self.ctx_len % 32]) & _MASK64
                pc = self.tc_pair.get(pk, 0) + 1; self.tc_pair[pk] = pc
                tot = self.tc_total.get(ck, 0) + 1; self.tc_total[ck] = tot
                if pc > self.tc_best_c.get(ck, 0): self.tc_best_c[ck] = pc; self.tc_best_t[ck] = tok
            self._push(tok)
            if not ib and not inw and self.wl > 0:
                wk = (self.wh ^ self.wl * _LEN_MIX) & _MASK64; wv[i] = True
                tot = self.wc_total.get(wk, 0)
                if tot > 0: wt[i], wp[i] = self.wc_best_t[wk], self.wc_best_c[wk] / tot
                wpk = (wk * _PAIR_MIX ^ tok * _ROLLING_COEFFS[0]) & _MASK64
                wpc = self.wc_pair.get(wpk, 0) + 1; self.wc_pair[wpk] = wpc
                tot = self.wc_total.get(wk, 0) + 1; self.wc_total[wk] = tot
                if wpc > self.wc_best_c.get(wk, 0): self.wc_best_c[wk] = wpc; self.wc_best_t[wk] = tok
                self.wh = (self.wh * _PREFIX_BASE ^ (tok + 1) * _ROLLING_COEFFS[self.wl % 32]) & _MASK64; self.wl += 1
            elif ib: self.wh = self.wl = 0
            else: self.wh = ((tok + 1) * _ROLLING_COEFFS[0]) & _MASK64; self.wl = 1
        return tt, tp, wt, wp, wv

class WordStartState:
    def __init__(self, *, sp, order, normalize_mode):
        self.sp, self.ctx_w, self.nm = sp, max(order - 1, 0), normalize_mode
        self.prev_ids: deque[int] = deque(maxlen=self.ctx_w)
        self.cur: list[int] = []
        self.w2id: dict[str, int] = {}; self.nid = 1
        self.ct: dict[tuple, int] = {}; self.pc: dict[tuple, int] = {}
        self.cbt: dict[tuple, int] = {}; self.cbc: dict[tuple, int] = {}
    def _flush(self):
        if not self.cur: return
        t = _normalize_word(self.sp.decode(self.cur), self.nm)
        if t:
            wid = self.w2id.get(t)
            if wid is None: wid = self.nid; self.w2id[t] = wid; self.nid += 1
            if self.ctx_w > 0: self.prev_ids.append(wid)
        self.cur = []
    def process_chunk(self, chunk_tokens, *, starts_new_word_lut, boundary_lut):
        ct = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        tt, tp = np.zeros(ct.size, dtype=np.uint16), np.zeros(ct.size, dtype=np.float32)
        for i, tu in enumerate(ct):
            tok = int(tu)
            if bool(boundary_lut[tok]): self._flush(); continue
            if bool(starts_new_word_lut[tok]): self._flush()
            ws = bool(starts_new_word_lut[tok]) or not self.cur
            ck = None
            if ws and len(self.prev_ids) >= self.ctx_w:
                ck = tuple(self.prev_ids) if self.ctx_w > 0 else ()
                tot = self.ct.get(ck, 0)
                if tot > 0: tt[i] = self.cbt[ck]; tp[i] = self.cbc[ck] / tot
            if ws:
                if ck is not None:
                    pk = (ck, tok); p = self.pc.get(pk, 0) + 1; self.pc[pk] = p
                    tot = self.ct.get(ck, 0) + 1; self.ct[ck] = tot
                    if p > self.cbc.get(ck, 0): self.cbc[ck] = p; self.cbt[ck] = tok
                self.cur = [tok]
            else: self.cur.append(tok)
        return tt, tp

def _piece_payload_bytes(sp, tid):
    if tid < 0 or tid >= sp.vocab_size(): return b""
    if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): return b""
    piece = sp.id_to_piece(tid)
    if sp.is_byte(tid):
        if piece.startswith("<0x") and piece.endswith(">"):
            try: return bytes([int(piece[3:-1], 16)])
            except ValueError: pass
        return sp.decode([tid]).encode("utf-8")
    return (piece[1:] if piece.startswith("\u2581") else piece).encode("utf-8")

class ByteContextTokenState:
    def __init__(self, *, order, token_payload_bytes, starts_new_word_lut, boundary_lut, seed_prefix_token):
        self.ctx_b = max(order, 0)
        self.tpb, self.snw, self.bnd = token_payload_bytes, starts_new_word_lut, boundary_lut
        self.prev: deque[int] = deque(maxlen=self.ctx_b if self.ctx_b > 0 else None)
        self.prev_bnd = True
        self.ct: dict[tuple, int] = {}; self.pc: dict[tuple, int] = {}
        self.cbt: dict[tuple, int] = {}; self.cbc: dict[tuple, int] = {}
        self._app(seed_prefix_token)
    def _app(self, tok):
        p = self.tpb[tok] if 0 <= tok < len(self.tpb) else b""
        if bool(self.snw[tok]) and not self.prev_bnd: p = b" " + p
        if self.ctx_b > 0:
            for b in p: self.prev.append(int(b))
        self.prev_bnd = bool(self.bnd[tok])
    def process_chunk(self, chunk_tokens):
        ct = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        tt, tp = np.zeros(ct.size, dtype=np.uint16), np.zeros(ct.size, dtype=np.float32)
        for i, tu in enumerate(ct):
            tok = int(tu)
            ck = () if self.ctx_b <= 0 else (tuple(self.prev) if len(self.prev) >= self.ctx_b else None)
            if ck is not None:
                tot = self.ct.get(ck, 0)
                if tot > 0: tt[i] = self.cbt[ck]; tp[i] = self.cbc[ck] / tot
                pk = (ck, tok); p = self.pc.get(pk, 0) + 1; self.pc[pk] = p
                tot = self.ct.get(ck, 0) + 1; self.ct[ck] = tot
                if p > self.cbc.get(ck, 0): self.cbc[ck] = p; self.cbt[ck] = tok
            self._app(tok)
        return tt, tp

class CTWTokenState:
    """Context Tree Weighting (Willems et al. 1995) — Bayesian multi-depth token predictor."""
    def __init__(self, *, depths=(1,2,3,4,6,8,12,16), vocab_size=1024, seed_prefix_token=0):
        self.depths, self.nd = sorted(depths), len(depths)
        self.hv = vocab_size / 2.0
        self.hist: list[int] = [seed_prefix_token]
        self.ct = [{} for _ in self.depths]; self.pc = [{} for _ in self.depths]
        self.cbt = [{} for _ in self.depths]; self.cbc = [{} for _ in self.depths]
    def _ch(self, d):
        n = len(self.hist)
        if n < d: return None
        h, b = 0, n - d
        for j in range(d): h ^= (self.hist[b + j] + 1) * (36313 + 27191 * j)
        return h & _MASK64
    def process_chunk(self, chunk_tokens):
        ct = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        n = int(ct.size)
        ot, op = np.zeros(n, dtype=np.uint16), np.zeros(n, dtype=np.float32)
        depths, nd, hv = self.depths, self.nd, self.hv
        ct_t, pc_t, cbt_t, cbc_t, hist = self.ct, self.pc, self.cbt, self.cbc, self.hist
        for i in range(n):
            tok = int(ct[i])
            pw, chosen, has = 0.0, 0, False
            for di in range(nd - 1, -1, -1):
                d = depths[di]
                if len(hist) < d: continue
                h = self._ch(d)
                if h is None: continue
                tot = ct_t[di].get(h, 0)
                kt = (0.5 / hv) if tot == 0 else ((cbc_t[di][h] + 0.5) / (tot + hv))
                tt = 0 if tot == 0 else cbt_t[di][h]
                if not has: pw, chosen, has = kt, tt, True
                else:
                    if tot > 0 and kt > pw: chosen = tt
                    pw = 0.5 * kt + 0.5 * pw
            if has: ot[i], op[i] = chosen, pw
            for di in range(nd):
                d = depths[di]
                if len(hist) < d: continue
                h = self._ch(d)
                if h is None: continue
                pk = (h << 16) ^ tok; p = pc_t[di].get(pk, 0) + 1; pc_t[di][pk] = p
                tot = ct_t[di].get(h, 0) + 1; ct_t[di][h] = tot
                if p > cbc_t[di].get(h, 0): cbc_t[di][h] = p; cbt_t[di][h] = tok
            hist.append(tok)
        return ot, op

class SuffixState:
    def __init__(self, *, order):
        self.cl = max(order, 1); self.cur: list[int] = []
        self.ct: dict[tuple, int] = {}; self.pc: dict[tuple, int] = {}
        self.cbt: dict[tuple, int] = {}; self.cbc: dict[tuple, int] = {}
    def process_chunk(self, chunk_tokens, *, starts_new_word_lut, boundary_lut):
        ct = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        tt, tp = np.zeros(ct.size, dtype=np.uint16), np.zeros(ct.size, dtype=np.float32)
        valid = np.zeros(ct.size, dtype=np.bool_)
        for i, tu in enumerate(ct):
            tok = int(tu)
            if bool(boundary_lut[tok]): self.cur = []; continue
            if bool(starts_new_word_lut[tok]): self.cur = []
            ck = tuple(self.cur[-min(len(self.cur), self.cl):]) if self.cur else None
            if ck is not None:
                tot = self.ct.get(ck, 0)
                if tot > 0: tt[i] = self.cbt[ck]; tp[i] = self.cbc[ck] / tot; valid[i] = True
                pk = (ck, tok); p = self.pc.get(pk, 0) + 1; self.pc[pk] = p
                tot = self.ct.get(ck, 0) + 1; self.ct[ck] = tot
                if p > self.cbc.get(ck, 0): self.cbc[ck] = p; self.cbt[ck] = tok
            self.cur.append(tok)
        return tt, tp, valid

def _build_piece_luts(tokenizer_path, vocab_size):
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    snw = np.zeros(vocab_size, dtype=np.uint8)
    tpb = [b""] * vocab_size
    for i in range(min(sp.vocab_size(), vocab_size)):
        snw[i] = 1 if sp.id_to_piece(i).startswith("\u2581") else 0
        tpb[i] = _piece_payload_bytes(sp, i)
    bnd = np.zeros(vocab_size, dtype=np.uint8)
    bid = sp.bos_id()
    if 0 <= bid < vocab_size: bnd[bid] = 1
    for t in range(min(sp.vocab_size(), vocab_size)):
        if sp.is_byte(t) and t in WHITESPACE_BYTE_IDS: bnd[t] = 1
    return sp, snw, bnd, tpb

def _np_slice(arr, c0, c1, dtype=np.uint16):
    return np.asarray(arr[c0:c1], dtype=dtype)

def eval_val_sliding_online_best_agree(
    *, args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride, batch_seqs=32, eval_seq_len=None, log0=print,
):
    t0 = time.perf_counter()
    seq_len = eval_seq_len or args.train_seq_len
    E = os.environ.get
    chunk_sz = int(E("CHUNK_TOKENS", "131072"))
    tok_order = int(E("TOKEN_ORDER", "16"))
    tok_thr = float(E("TOKEN_THRESHOLD", "0.800")); tok_bst = float(E("TOKEN_BOOST", "2.625"))
    ctw_on = bool(int(E("CTW_ENABLED", "1")))
    ctw_depths = tuple(int(x) for x in E("CTW_DEPTHS", "1,2,3,4,6,8,12,16").split(",") if x.strip())
    ctw_tau = float(E("CTW_TAU", "0.003")); ctw_bst = float(E("CTW_BOOST", "2.625"))
    w_tau = float(E("WITHIN_TAU", "0.450")); w_bst = float(E("WITHIN_BOOST", "0.750"))
    by_ord = int(E("BYTE_ORDER", "24")); by_tau = float(E("BYTE_TAU", "0.550")); by_bst = float(E("BYTE_BOOST", "0.900"))
    wo_ord = int(E("WORD_ORDER", "4")); wo_nm = E("WORD_NORMALIZE", "strip_punct_lower")
    wo_tau = float(E("WORD_TAU", "0.650")); wo_bst = float(E("WORD_BOOST", "0.750"))
    sx_ord = int(E("SUFFIX_ORDER", "4")); sx_tau = float(E("SUFFIX_TAU", "0.425")); sx_bst = float(E("SUFFIX_BOOST", "0.650"))
    aa_bst = float(E("AGREE_ADD_BOOST", "0.500")); min_eg = float(E("MIN_EXPECTED_GAIN", "0.0"))
    ds = float(E("DISAGREE_SCALE", "0.35"))
    ah_sz = int(E("AGREE_HISTORY_SIZE", "16384")); ah_pct = float(E("AGREE_PERCENTILE", "0.80"))
    ah_min = int(E("AGREE_MIN_HISTORY", "1024")); ew_pow = float(E("EXPERT_WEIGHT_POWER", "2.0"))

    total_tgt = val_tokens.numel() - 1
    tok_np = val_tokens.cpu().numpy().astype(np.uint16, copy=False)
    sp, snw_lut, bnd_lut, tpb_list = _build_piece_luts(args.tokenizer_path, args.vocab_size)
    seed = int(tok_np[0])
    ngram = OnlineNgramState(token_ctx_len=max(tok_order - 1, 0), starts_new_word_lut=snw_lut, boundary_lut=bnd_lut, seed_prefix_token=seed)
    word_st = WordStartState(sp=sp, order=wo_ord, normalize_mode=wo_nm)
    byte_st = ByteContextTokenState(order=by_ord, token_payload_bytes=tpb_list, starts_new_word_lut=snw_lut, boundary_lut=bnd_lut, seed_prefix_token=seed)
    sfx_st = SuffixState(order=sx_ord)
    ctw_st = CTWTokenState(depths=ctw_depths, vocab_size=args.vocab_size, seed_prefix_token=seed) if ctw_on else None
    expert_names = ["ctw", "token", "within", "byte", "word", "suffix"]
    estats = {n: {"uses": 0.0, "hits": 0.0} for n in expert_names}
    gain_hist: deque[float] = deque(maxlen=max(ah_sz, 1))
    _log = log0 if rank == 0 else (lambda *_: None)
    compiled = _compile_logits_fn(base_model, seq_len=seq_len, device=device, log0=_log)
    su_s = _dist_max(time.perf_counter() - t0, device, world_size)
    if rank == 0:
        log0(f"online_best_agree:start total_targets={total_tgt} seq_len={seq_len} stride={stride} "
             f"chunk_tokens={chunk_sz} ctw_enabled={ctw_on} startup_max={su_s:.2f}s")
    cws = _build_chunk_windows(total_tgt, seq_len, stride, chunk_sz)
    llm_ls = ba_ls = by_s = tc = st_t = in_t = fw_t = bl_t = 0.0
    loop_t0 = time.perf_counter()
    try:
      with torch.inference_mode():
        for ci, windows in enumerate(cws):
            if not windows: continue
            ct0 = ci * chunk_sz; ct1 = min((ci + 1) * chunk_sz, total_tgt)
            ctgt = np.ascontiguousarray(tok_np[ct0 + 1 : ct1 + 1], dtype=np.uint16)
            ts0 = time.perf_counter()
            ttt, ttp, wtt, wtp, wv = ngram.process_chunk(ctgt)
            ctt = cpp = None
            if ctw_st: ctt, cpp = ctw_st.process_chunk(ctgt)
            btt, btp = byte_st.process_chunk(ctgt)
            wott, wotp = word_st.process_chunk(ctgt, starts_new_word_lut=snw_lut, boundary_lut=bnd_lut)
            stt, stp, sv = sfx_st.process_chunk(ctgt, starts_new_word_lut=snw_lut, boundary_lut=bnd_lut)
            adapt_thr = min_eg
            if len(gain_hist) >= ah_min:
                adapt_thr = max(min_eg, float(np.quantile(np.asarray(gain_hist, dtype=np.float32), ah_pct)))
            ew = {n: _expert_rel_weight(s["hits"], s["uses"], ew_pow) for n, s in estats.items()}
            st_t += time.perf_counter() - ts0
            epreds = [("token", ttt, ttp, tok_thr, tok_bst, None), ("within", wtt, wtp, w_tau, w_bst, wv),
                      ("byte", btt, btp, by_tau, by_bst, None), ("word", wott, wotp, wo_tau, wo_bst, None),
                      ("suffix", stt, stp, sx_tau, sx_bst, sv)]
            if ctw_st: epreds.insert(0, ("ctw", ctt, cpp, ctw_tau, ctw_bst, None))
            start_end = lambda r, ws: ((len(windows) * r) // ws, (len(windows) * (r + 1)) // ws)
            s_e = start_end(rank, world_size)
            my_wins = windows[s_e[0]:s_e[1]]
            for bi in range(0, len(my_wins), batch_seqs):
                bws = my_wins[bi:bi + batch_seqs]
                if not bws: continue
                ti0 = time.perf_counter()
                bsz = len(bws)
                x_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                e_batches = {nm: torch.zeros(bsz, seq_len, dtype=torch.int64, device=device) for nm, *_ in epreds}
                wlens, sstarts = [], []
                for i, ws in enumerate(bws):
                    end = min(ws + seq_len, total_tgt); wlen = end - ws; wlens.append(wlen)
                    loc = val_tokens[ws:end + 1].to(device=device, dtype=torch.int64)
                    x_b[i, :wlen] = loc[:-1]; y_b[i, :wlen] = loc[1:]
                    s = 0 if ws == 0 else max(wlen - stride, 0); sstarts.append(s)
                    if wlen - s <= 0: continue
                    c0, c1 = ws + s - ct0, ws + wlen - ct0
                    for nm, tt, *_ in epreds:
                        e_batches[nm][i, s:wlen] = torch.from_numpy(np.asarray(tt[c0:c1], dtype=np.int64)).to(device)
                in_t += time.perf_counter() - ti0
                tf0 = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if hasattr(torch.compiler, "cudagraph_mark_step_begin"): torch.compiler.cudagraph_mark_step_begin()
                    logits = compiled(x_b)
                lf = logits.float(); ln = torch.logsumexp(lf, dim=-1)
                true_p = (lf.gather(-1, y_b.unsqueeze(-1)).squeeze(-1) - ln).exp()
                e_hints = {nm: (lf.gather(-1, eb.unsqueeze(-1)).squeeze(-1) - ln).exp() for nm, eb in e_batches.items()}
                fw_t += time.perf_counter() - tf0
                tb0 = time.perf_counter()
                for i, ws in enumerate(bws):
                    wlen, s = wlens[i], sstarts[i]
                    if wlen - s <= 0: continue
                    c0, c1 = ws + s - ct0, ws + wlen - ct0
                    llm_c = true_p[i, s:wlen].detach().cpu().numpy().astype(np.float64, copy=False)
                    experts = []
                    for nm, ett, etp, tau, bst, vmask in epreds:
                        hint = e_hints[nm][i, s:wlen].detach().cpu().numpy().astype(np.float32, copy=False)
                        g = _np_slice(etp, c0, c1, np.float32) >= tau
                        if vmask is not None: g = _np_slice(vmask, c0, c1, np.bool_) & g
                        experts.append({"top_token": _np_slice(ett, c0, c1), "top_prob": _np_slice(etp, c0, c1, np.float32),
                                        "hint_probs": hint, "gate": g, "boost": bst, "weight": ew[nm]})
                    ba_c, cs, csup = _compute_best_agree(llm_chunk=llm_c, true_targets=ctgt[c0:c1],
                        experts=experts, agree_add_boost=aa_bst, min_eg=adapt_thr, disagree_scale=ds)
                    if csup.any(): gain_hist.extend(cs[csup].astype(np.float64).tolist())
                    llm_ls += float((-np.log(np.clip(llm_c, 1e-12, 1.0))).sum())
                    ba_ls += float((-np.log(np.clip(ba_c, 1e-12, 1.0))).sum())
                    tc += float(c1 - c0)
                    tgt, prev = y_b[i, s:wlen], x_b[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    by_s += float(tb.sum().item())
                bl_t += time.perf_counter() - tb0
            for nm, ett, etp, tau, _, vmask in epreds:
                g = (np.asarray(etp, dtype=np.float32) >= tau)
                if vmask is not None: g = np.asarray(vmask, dtype=np.bool_) & g
                estats[nm]["uses"] += float(g.sum())
                estats[nm]["hits"] += float((g & (np.asarray(ett, dtype=np.uint16) == ctgt)).sum())
    finally:
        ngram.close()
    def _red(v): return _dist_max(v, device, world_size)
    _f64 = lambda v: torch.tensor([v], dtype=torch.float64, device=device)
    ll_t, bl_t2, bs_t, tc_t = _f64(llm_ls), _f64(ba_ls), _f64(by_s), _f64(tc)
    if world_size > 1:
        for t in [ll_t, bl_t2, bs_t, tc_t]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    llm_tl, ba_tl = float(ll_t.item()), float(bl_t2.item())
    tb, ttc = float(bs_t.item()), float(tc_t.item())
    l2b = lambda tl: tl / (tb * math.log(2.0))
    llm_bpb, ba_bpb = l2b(llm_tl), l2b(ba_tl)
    timings = {"llm_bpb": llm_bpb, "best_agree_bpb": ba_bpb, "gain_bpb": llm_bpb - ba_bpb,
               "startup_max_s": su_s, "loop_total_max_s": _red(time.perf_counter() - loop_t0),
               "state_max_s": _red(st_t), "input_max_s": _red(in_t), "forward_max_s": _red(fw_t), "blend_max_s": _red(bl_t),
               "llm_nats_per_byte": llm_tl / tb, "best_agree_nats_per_byte": ba_tl / tb,
               "gain_nats_per_byte": (llm_tl - ba_tl) / tb}
    if rank == 0:
        log0(f"online_best_agree:done llm_bpb={llm_bpb:.8f} best_agree_bpb={ba_bpb:.8f} "
             f"gain_bpb={llm_bpb - ba_bpb:.8f} startup_max={su_s:.2f}s "
             f"loop_total_max={timings['loop_total_max_s']:.2f}s")
    return ba_tl / max(ttc, 1.0), ba_bpb, timings

def _compile_logits_fn(model, *, seq_len, device, log0):
    if os.environ.get("EVAL_COMPILE", "0") != "1":
        log0("eval-pass-online: using eager logits path"); return model.forward_logits
    log0("eval-pass-online: compiling logits path")
    compiled = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)
    dummy = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"): torch.compiler.cudagraph_mark_step_begin()
        _ = compiled(dummy)
    del dummy; log0("eval-pass-online: compile warmup done"); return compiled

# --- Training ---

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    logfile = None
    record_logfiles: list[str] = []
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        record_logfiles.append(f"train_seed{args.seed}.log")
        if args.seed == 1337:
            record_logfiles.append("train.log")
        for record_logfile in record_logfiles:
            Path(record_logfile).write_text("", encoding="utf-8")
        print(logfile)
    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)
        for record_logfile in record_logfiles:
            with open(record_logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        asym_logit_rescale=args.asym_logit_rescale,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        bigram_hash_mode=args.bigram_hash_mode,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        neg_slope=args.negative_slope,
        attention_sink=args.attention_sink,
        expert_mixer_enabled=args.expert_mixer_enabled,
        expert_recent_window=args.expert_recent_window,
        expert_cache_window=args.expert_cache_window,
        expert_cache_dim=args.expert_cache_dim,
    ).to(device).bfloat16()
    base_model.qo_bank.data = base_model.qo_bank.data.float()
    base_model.kv_bank.data = base_model.kv_bank.data.float()
    base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
    base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    prime_rotary_caches(base_model, device, args.train_seq_len, effective_eval_seq_len)
    compiled_model = compile_with_env(base_model)
    model = compiled_model

    matrix_params = [
        base_model.qo_bank, base_model.kv_bank,
        base_model.mlp_up_bank, base_model.mlp_down_bank,
    ]
    block_named_params = list(base_model.blocks.named_parameters())
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if getattr(base_model, "asym_logit_rescale", False):
        scalar_params.append(base_model.softcap_pos)
        scalar_params.append(base_model.softcap_neg)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            scalar_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            scalar_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)
    if base_model.expert_mixer is not None:
        scalar_params.extend(list(base_model.expert_mixer.parameters()))
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    replicated_params = list(optimizer_tok.param_groups[0]["params"])
    for pg in optimizer_tok.param_groups[1:]:
        replicated_params.extend(pg["params"])
    replicated_params.extend(scalar_params)

    optimizer_head = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        replicated_params.append(base_model.lm_head.weight)
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if optimizer_head is not None:
        optimizers.append(optimizer_head)
    log0(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"torch_compile:disable={TORCH_COMPILE_DISABLE} backend:{TORCH_COMPILE_BACKEND} "
        f"mode:{TORCH_COMPILE_MODE or 'default'} dynamic:{TORCH_COMPILE_DYNAMIC} "
        f"fullgraph:{TORCH_COMPILE_FULLGRAPH}"
    )
    log0(
        f"torch_compile_eval:disable={TORCH_COMPILE_EVAL_DISABLE} "
        f"mode:{TORCH_COMPILE_EVAL_MODE or 'inherit'}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"swa:enabled={args.swa_enabled} every:{args.swa_every} "
        f"start_step:{args.swa_start_step or 'auto'} start_scale:{args.swa_start_scale:.3f}"
    )
    log0(f"ema:decay_start:{args.ema_start_decay:.4f} decay_end:{args.ema_end_decay:.4f}")
    log0(f"attention_sink:enabled={args.attention_sink}")
    log0(
        f"expert_mixer:enabled={args.expert_mixer_enabled} "
        f"recent_window:{args.expert_recent_window} "
        f"cache_window:{args.expert_cache_window} cache_dim:{args.expert_cache_dim}"
    )
    log0(f"bigram_hash:mode:{args.bigram_hash_mode} vocab:{args.bigram_vocab_size} dim:{args.bigram_dim}")
    log0(
        f"gptq:block_size:{args.gptq_block_size} percdamp:{args.gptq_percdamp:.4f} "
        f"int8_attn_layers:{sorted(parse_layer_spec(args.int8_attn_layers, args.num_layers))} "
        f"int8_mlp_layers:{sorted(parse_layer_spec(args.int8_mlp_layers, args.num_layers))} "
        f"lqer_enabled:{args.lqer_enabled} lqer_rank:{args.lqer_rank} "
        f"lqer_top_k:{args.lqer_top_k} lqer_asym_group:{args.lqer_asym_group}"
    )
    log0(f"ttt_online_agree_order:{args.ttt_online_agree_order}")
    log0(
        f"lr_schedule:warmdown_iters:{args.warmdown_iters} "
        f"step_ms_ref:{args.lr_schedule_reference_step_ms or 'auto'}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    training_budget_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    max_wallclock_ms = training_budget_ms
    if args.use_gptq and training_budget_ms is not None:
        max_wallclock_ms = max(training_budget_ms - args.gptq_reserve_ms, 0.0)
        log0(
            f"gptq:reserving {args.gptq_reserve_ms:.0f}ms from training budget, "
            f"loop_budget={max_wallclock_ms:.0f}ms total_budget={training_budget_ms:.0f}ms"
        )
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        if args.lr_schedule_reference_step_ms > 0:
            step_ms = max(step_ms, args.lr_schedule_reference_step_ms)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    def ema_decay_for_step(step: int) -> float:
        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        if warmdown_start <= 0:
            return args.ema_end_decay
        frac = min(step / warmdown_start, 1.0)
        return args.ema_start_decay + (args.ema_end_decay - args.ema_start_decay) * frac
    training_time_ms = 0.0
    if args.warmup_steps > 0:
        torch.cuda.synchronize()
        t_warmup = time.perf_counter()
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    cudagraph_step_begin()
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        torch.cuda.synchronize()
        training_time_ms += 1000.0 * (time.perf_counter() - t_warmup)
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                cudagraph_step_begin()
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        optimizer_muon.launch_reduce_scatters()
        if distributed:
            for p in replicated_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        optimizer_tok.step()
        optimizer_scalar.step()
        if optimizer_head is not None:
            optimizer_head.step()
        optimizer_muon.step()
        zero_grad_all()
        ema_decay = ema_decay_for_step(step + 1)
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        swa_ready = step >= args.swa_start_step if args.swa_start_step > 0 else scale < args.swa_start_scale
        if args.swa_enabled and swa_ready and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    log0("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    gptq_hessians = None
    if args.use_gptq:
        torch.cuda.synchronize()
        t_gptq = time.perf_counter()
        log0(
            f"gptq:calibrating with {args.gptq_calib_samples} batches (training data, "
            "counted inside training budget)..."
        )
        calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        gptq_hessians = gptq_collect_hessians(
            base_model, calib_loader, device, num_batches=args.gptq_calib_samples,
            batch_tokens=args.train_batch_tokens, seq_len=args.train_seq_len,
            grad_accum_steps=grad_accum_steps)
        del calib_loader
        torch.cuda.synchronize()
        gptq_elapsed_ms = 1000.0 * (time.perf_counter() - t_gptq)
        training_time_ms += gptq_elapsed_ms
        log0(
            f"gptq:calibrated {len(gptq_hessians)} layers in {gptq_elapsed_ms / 1000.0:.1f}s "
            f"counted_training_time:{training_time_ms:.0f}ms"
        )
        if training_budget_ms is not None:
            remaining_ms = max(training_budget_ms - training_time_ms, 0.0)
            log0(
                f"training_budget:used={training_time_ms:.0f}ms "
                f"budget={training_budget_ms:.0f}ms remaining={remaining_ms:.0f}ms"
            )
            if training_time_ms > training_budget_ms:
                raise RuntimeError(
                    f"GPTQ calibration exceeded the counted training budget: "
                    f"{training_time_ms:.0f}ms > {training_budget_ms:.0f}ms"
                )
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )
    export_sd = base_model.state_dict()
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = submission_code_bytes()
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
    int8_attn_layers = parse_layer_spec(args.int8_attn_layers, args.num_layers)
    int8_mlp_layers = parse_layer_spec(args.int8_mlp_layers, args.num_layers)
    quant_result, quant_meta = mixed_quantize_int6(
        unbanked_sd,
        {"mlp", "attn"},
        clip_range=args.quant_clip_range,
        hessians=gptq_hessians,
        gptq_block_size=args.gptq_block_size,
        gptq_percdamp=args.gptq_percdamp,
        int8_attn_layers=int8_attn_layers,
        int8_mlp_layers=int8_mlp_layers,
        lqer_enabled=args.lqer_enabled,
        lqer_rank=args.lqer_rank,
        lqer_top_k=args.lqer_top_k,
        lqer_asym_group=args.lqer_asym_group,
    )
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=6)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = submission_code_bytes()
        log0(f"Serialized model int6+lzma: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+lzma: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(lzma.decompress(quant_blob_disk)),
        map_location="cpu",
    )
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, asym_logit_rescale=args.asym_logit_rescale,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        bigram_hash_mode=args.bigram_hash_mode,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        neg_slope=args.negative_slope,
        attention_sink=args.attention_sink,
        expert_mixer_enabled=args.expert_mixer_enabled,
        expert_recent_window=args.expert_recent_window,
        expert_cache_window=args.expert_cache_window,
        expert_cache_dim=args.expert_cache_dim,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    prime_rotary_caches(eval_model, device, effective_eval_seq_len)
    compiled_eval = compile_with_env(eval_model, is_eval=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    log0(
        f"DIAGNOSTIC delta post_ema_to_int6_quant_tax_bpb:{q_val_bpb - diag_val_bpb:.8f} "
        f"post_ema_bpb:{diag_val_bpb:.8f} int6_bpb:{q_val_bpb:.8f}"
    )
    sw_seq_len = effective_eval_seq_len
    sliding_baseline_bpb = q_val_bpb
    online_best_agree_enabled = bool(int(os.environ.get("ONLINE_BEST_AGREE_EVAL", "0")))
    online_full_vocab_mix_enabled = bool(int(os.environ.get("ONLINE_FULL_VOCAB_MIX_EVAL", "1")))
    prefix_mix_ttt_enabled = bool(int(os.environ.get("PREFIX_MIX_TTT_EVAL", "1")))
    online_best_agree_order = args.ttt_online_agree_order
    if online_best_agree_order not in {"ttt_first", "overlay_first", "both"}:
        online_best_agree_order = "ttt_first"
    def run_online_best_agree(stage_suffix: str = "") -> None:
        if not online_best_agree_enabled:
            return
        torch.cuda.synchronize()
        t_online = time.perf_counter()
        online_val_loss, online_val_bpb, online_timings = eval_val_sliding_online_best_agree(
            args=args,
            base_model=eval_model,
            rank=rank,
            world_size=world_size,
            device=device,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            stride=args.eval_stride,
            batch_seqs=int(os.environ.get("BATCH_SEQS", "32")),
            eval_seq_len=sw_seq_len,
            log0=log0,
        )
        torch.cuda.synchronize()
        online_elapsed_ms = 1000.0 * (time.perf_counter() - t_online)
        metric_prefix = "online_best_agree" if not stage_suffix else f"online_best_agree_{stage_suffix}"
        window_prefix = "online_best_agree_sliding_window" if not stage_suffix else f"{metric_prefix}_sliding_window"
        compare_prefix = "online_best_agree_compare" if not stage_suffix else f"{metric_prefix}_compare"
        timing_prefix = "online_best_agree_timing" if not stage_suffix else f"{metric_prefix}_timing"
        log0(
            f"{window_prefix} val_loss:{online_val_loss:.4f} "
            f"val_bpb:{online_val_bpb:.4f} stride:{args.eval_stride} eval_time:{online_elapsed_ms:.0f}ms"
        )
        log0(
            f"{window_prefix}_exact val_loss:{online_val_loss:.8f} "
            f"val_bpb:{online_val_bpb:.8f}"
        )
        log0(
            f"{compare_prefix} "
            f"llm_bpb:{online_timings['llm_bpb']:.8f} "
            f"best_agree_bpb:{online_timings['best_agree_bpb']:.8f} "
            f"gain_bpb:{online_timings['gain_bpb']:.8f} "
            f"llm_nats_per_byte:{online_timings['llm_nats_per_byte']:.8f} "
            f"best_agree_nats_per_byte:{online_timings['best_agree_nats_per_byte']:.8f} "
            f"gain_nats_per_byte:{online_timings['gain_nats_per_byte']:.8f}"
        )
        log0(
            f"{timing_prefix} "
            f"startup_max:{online_timings['startup_max_s']:.2f}s "
            f"loop_total_max:{online_timings['loop_total_max_s']:.2f}s "
            f"state_max:{online_timings['state_max_s']:.2f}s "
            f"input_max:{online_timings['input_max_s']:.2f}s "
            f"forward_max:{online_timings['forward_max_s']:.2f}s "
            f"blend_max:{online_timings['blend_max_s']:.2f}s "
            f"wallclock:{online_elapsed_ms / 1000.0:.2f}s"
        )
        log0(
            f"DIAGNOSTIC delta {metric_prefix}_gain_from_sliding_bpb:"
            f"{sliding_baseline_bpb - online_val_bpb:.8f}"
        )
    def run_online_full_vocab_mix(stage_suffix: str = "") -> None:
        if not online_full_vocab_mix_enabled:
            return
        torch.cuda.synchronize()
        t_online = time.perf_counter()
        online_val_loss, online_val_bpb, online_timings = eval_val_sliding_online_full_vocab_mix(
            args=args,
            base_model=eval_model,
            rank=rank,
            world_size=world_size,
            device=device,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            stride=args.eval_stride,
            batch_seqs=int(os.environ.get("BATCH_SEQS", "32")),
            eval_seq_len=sw_seq_len,
            log0=log0,
        )
        torch.cuda.synchronize()
        online_elapsed_ms = 1000.0 * (time.perf_counter() - t_online)
        metric_prefix = "online_full_vocab_mix" if not stage_suffix else f"online_full_vocab_mix_{stage_suffix}"
        window_prefix = f"{metric_prefix}_sliding_window"
        compare_prefix = f"{metric_prefix}_compare"
        timing_prefix = f"{metric_prefix}_timing"
        log0(
            f"{window_prefix} val_loss:{online_val_loss:.4f} "
            f"val_bpb:{online_val_bpb:.4f} stride:{args.eval_stride} eval_time:{online_elapsed_ms:.0f}ms"
        )
        log0(
            f"{window_prefix}_exact val_loss:{online_val_loss:.8f} "
            f"val_bpb:{online_val_bpb:.8f}"
        )
        log0(
            f"{compare_prefix} "
            f"llm_bpb:{online_timings['llm_bpb']:.8f} "
            f"mix_bpb:{online_timings['mix_bpb']:.8f} "
            f"gain_bpb:{online_timings['gain_bpb']:.8f} "
            f"llm_nats_per_byte:{online_timings['llm_nats_per_byte']:.8f} "
            f"mix_nats_per_byte:{online_timings['mix_nats_per_byte']:.8f} "
            f"gain_nats_per_byte:{online_timings['gain_nats_per_byte']:.8f} "
            f"use_rate:{online_timings['cache_use_rate']:.6f} "
            f"hit_rate:{online_timings['cache_hit_rate']:.6f} "
            f"avg_lambda:{online_timings['avg_lambda']:.6f} "
            f"avg_top_prob_when_used:{online_timings['avg_top_prob_when_used']:.6f}"
        )
        log0(
            f"{timing_prefix} "
            f"startup_max:{online_timings['startup_max_s']:.2f}s "
            f"loop_total_max:{online_timings['loop_total_max_s']:.2f}s "
            f"state_max:{online_timings['state_max_s']:.2f}s "
            f"input_max:{online_timings['input_max_s']:.2f}s "
            f"forward_max:{online_timings['forward_max_s']:.2f}s "
            f"blend_max:{online_timings['blend_max_s']:.2f}s "
            f"wallclock:{online_elapsed_ms / 1000.0:.2f}s"
        )
        log0(
            f"DIAGNOSTIC delta {metric_prefix}_gain_from_sliding_bpb:"
            f"{sliding_baseline_bpb - online_val_bpb:.8f}"
        )
    def run_prefix_mix_ttt() -> None:
        torch.cuda.synchronize()
        t_online = time.perf_counter()
        online_val_loss, online_val_bpb, online_timings = eval_val_sliding_prefix_mix_ttt(
            args=args,
            base_model=eval_model,
            rank=rank,
            world_size=world_size,
            device=device,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            stride=args.eval_stride,
            batch_seqs=int(os.environ.get("BATCH_SEQS", "32")),
            eval_seq_len=sw_seq_len,
            log0=log0,
        )
        torch.cuda.synchronize()
        online_elapsed_ms = 1000.0 * (time.perf_counter() - t_online)
        log0(
            f"legal_prefix_mix_ttt_sliding_window val_loss:{online_val_loss:.4f} "
            f"val_bpb:{online_val_bpb:.4f} stride:{args.eval_stride} eval_time:{online_elapsed_ms:.0f}ms"
        )
        log0(
            f"legal_prefix_mix_ttt_sliding_window_exact val_loss:{online_val_loss:.8f} "
            f"val_bpb:{online_val_bpb:.8f}"
        )
        log0(
            "legal_prefix_mix_ttt_compare "
            f"llm_bpb:{online_timings['llm_bpb']:.8f} "
            f"mix_bpb:{online_timings['mix_bpb']:.8f} "
            f"gain_bpb:{online_timings['gain_bpb']:.8f} "
            f"use_rate:{online_timings['cache_use_rate']:.6f} "
            f"hit_rate:{online_timings['cache_hit_rate']:.6f} "
            f"avg_lambda:{online_timings['avg_lambda']:.6f} "
            f"avg_top_prob_when_used:{online_timings['avg_top_prob_when_used']:.6f}"
        )
        log0(
            "legal_prefix_mix_ttt_timing "
            f"startup_max:{online_timings['startup_max_s']:.2f}s "
            f"loop_total_max:{online_timings['loop_total_max_s']:.2f}s "
            f"state_max:{online_timings['state_max_s']:.2f}s "
            f"input_max:{online_timings['input_max_s']:.2f}s "
            f"forward_max:{online_timings['forward_max_s']:.2f}s "
            f"blend_max:{online_timings['blend_max_s']:.2f}s "
            f"train_max:{online_timings['train_max_s']:.2f}s "
            f"wallclock:{online_elapsed_ms / 1000.0:.2f}s"
        )
        log0(
            "DIAGNOSTIC delta legal_prefix_mix_ttt_gain_from_sliding_bpb:"
            f"{sliding_baseline_bpb - online_val_bpb:.8f}"
        )
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        log0(
            f"DIAGNOSTIC delta int6_to_sliding_gain_bpb:{q_val_bpb - sw_val_bpb:.8f} "
            f"int6_bpb:{q_val_bpb:.8f} sliding_bpb:{sw_val_bpb:.8f}"
        )
        sliding_baseline_bpb = sw_val_bpb
    if args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize()
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
            f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms"
        )
        log0(f"final_int6_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
    can_run_online = args.eval_stride > 0 and args.eval_stride < sw_seq_len and online_best_agree_enabled
    can_run_full_vocab_mix = args.eval_stride > 0 and args.eval_stride < sw_seq_len and online_full_vocab_mix_enabled
    can_run_prefix_mix_ttt = args.ttt_enabled and can_run_full_vocab_mix and prefix_mix_ttt_enabled
    if can_run_full_vocab_mix and args.ttt_enabled and not can_run_prefix_mix_ttt and online_best_agree_order in {"overlay_first", "both"}:
        run_online_full_vocab_mix("pre_ttt")
    if can_run_online and args.ttt_enabled and not can_run_prefix_mix_ttt and online_best_agree_order in {"overlay_first", "both"}:
        run_online_best_agree("pre_ttt")
    if args.ttt_enabled:
        if can_run_prefix_mix_ttt:
            run_prefix_mix_ttt()
        else:
            torch.cuda.synchronize()
            t_ttt = time.perf_counter()
            ttt_loss, ttt_bpb = eval_val_sliding_ttt(
                args, eval_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=args.eval_stride, log0=log0,
            )
            torch.cuda.synchronize()
            log0(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
                 f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")
            log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
    if can_run_online:
        if not args.ttt_enabled:
            run_online_best_agree()
    if can_run_full_vocab_mix:
        if not args.ttt_enabled:
            run_online_full_vocab_mix()
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
