#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path


FLOAT_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
DEFAULT_DATA_PATH = "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = "/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
DEFAULT_FOLDER = (
    "/workspace/parameter-golf/records/track_10min_16mb/"
    "2026-04-02_11L_XSA4_PartialRoPE_LNScale_VE128_ASQU_EMA_GPTQlite"
)

RE_SEED = re.compile(r"\bseed:(\d+)\b")
RE_STEP_TRAIN = re.compile(rf"step:(\d+)/(\d+) train_loss:{FLOAT_RE} .* step_avg:({FLOAT_RE})ms")
RE_STEP_VAL = re.compile(rf"step:(\d+)/(\d+) val_loss:({FLOAT_RE}) val_bpb:({FLOAT_RE}).*step_avg:({FLOAT_RE})ms")
RE_DIAGNOSTIC = re.compile(rf"DIAGNOSTIC post_(ema|swa|raw) val_loss:({FLOAT_RE}) val_bpb:({FLOAT_RE})")
RE_BEST_AVG = re.compile(rf"BEST_AVG:(\w+) val_bpb:({FLOAT_RE})")
RE_ROUNDTRIP = re.compile(rf"final_int6_roundtrip_exact val_loss:({FLOAT_RE}) val_bpb:({FLOAT_RE})")
RE_SLIDING = re.compile(rf"final_int6_sliding_window_exact val_loss:({FLOAT_RE}) val_bpb:({FLOAT_RE})")
RE_TOTAL_SIZE = re.compile(r"Total submission size [^:]+: (\d+) bytes")
RE_ACTIVATION = re.compile(
    rf"mlp_activation:(\S+) mlp_leaky_slope:({FLOAT_RE}) "
    rf"asqu_beta_init:({FLOAT_RE}) asqu_lr:({FLOAT_RE})(?: asqu_sigmoid_beta:(\S+))?"
)
RE_QAT_AND_VE = re.compile(rf"late_qat_threshold:({FLOAT_RE}) ve_enabled:(\S+) ve_layers:(.+)")
RE_XSA = re.compile(r"XSA:last_(\d+) active_layers:\[(.*?)\]")
RE_ASQU_DIAG = re.compile(
    rf"asqu_diag layer:(\d+) beta_mean:({FLOAT_RE}) beta_std:({FLOAT_RE}) "
    rf"beta_min:({FLOAT_RE}) beta_max:({FLOAT_RE}) "
    rf"frac_near_zero:({FLOAT_RE}) frac_gt_half:({FLOAT_RE})"
)
RE_QUANT_DIAG = re.compile(rf"quant_diag (\S+) int6_mse:({FLOAT_RE}) shape:(\[.*\])")


@dataclass
class QuantDiag:
    name: str
    mse: float
    shape: str


@dataclass
class AsquDiag:
    layer: int
    beta_mean: float
    beta_std: float
    beta_min: float
    beta_max: float
    frac_near_zero: float
    frac_gt_half: float


@dataclass
class RunMetrics:
    path: str
    label: str
    seed: int | None = None
    mlp_activation: str | None = None
    mlp_leaky_slope: float | None = None
    asqu_beta_init: float | None = None
    asqu_lr: float | None = None
    asqu_sigmoid_beta: bool | None = None
    late_qat_threshold: float | None = None
    ve_enabled: bool | None = None
    ve_layers: str | None = None
    xsa_last_n: int | None = None
    xsa_active_layers: list[int] = field(default_factory=list)
    end_train_val_loss: float | None = None
    end_train_val_bpb: float | None = None
    step_avg_ms: float | None = None
    diag_post_ema_val_bpb: float | None = None
    diag_post_swa_val_bpb: float | None = None
    diag_post_raw_val_bpb: float | None = None
    best_avg_name: str | None = None
    best_avg_val_bpb: float | None = None
    final_int6_roundtrip_exact_val_bpb: float | None = None
    final_int6_sliding_window_exact_val_bpb: float | None = None
    total_submission_size_bytes: int | None = None
    quant_diags: dict[str, QuantDiag] = field(default_factory=dict)
    asqu_diags: list[AsquDiag] = field(default_factory=list)

    def prequant_metric(self) -> tuple[float | None, str]:
        if self.best_avg_val_bpb is not None:
            return self.best_avg_val_bpb, "BEST_AVG"
        if self.diag_post_ema_val_bpb is not None:
            return self.diag_post_ema_val_bpb, "post_ema"
        if self.end_train_val_bpb is not None:
            return self.end_train_val_bpb, "end_of_train"
        return None, "unavailable"


def parse_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in {"1", "true", "yes"}:
        return True
    if norm in {"0", "false", "no"}:
        return False
    return None


def parse_active_layers(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_log(path: Path, label: str | None = None) -> RunMetrics:
    metrics = RunMetrics(path=str(path), label=label or path.stem)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if match := RE_SEED.search(line):
                metrics.seed = int(match.group(1))
            if match := RE_STEP_TRAIN.search(line):
                metrics.step_avg_ms = float(match.group(3))
            if match := RE_STEP_VAL.search(line):
                metrics.end_train_val_loss = float(match.group(3))
                metrics.end_train_val_bpb = float(match.group(4))
                metrics.step_avg_ms = float(match.group(5))
            if match := RE_DIAGNOSTIC.search(line):
                which = match.group(1)
                value = float(match.group(3))
                if which == "ema":
                    metrics.diag_post_ema_val_bpb = value
                elif which == "swa":
                    metrics.diag_post_swa_val_bpb = value
                elif which == "raw":
                    metrics.diag_post_raw_val_bpb = value
            if match := RE_BEST_AVG.search(line):
                metrics.best_avg_name = match.group(1)
                metrics.best_avg_val_bpb = float(match.group(2))
            if match := RE_ROUNDTRIP.search(line):
                metrics.final_int6_roundtrip_exact_val_bpb = float(match.group(2))
            if match := RE_SLIDING.search(line):
                metrics.final_int6_sliding_window_exact_val_bpb = float(match.group(2))
            if match := RE_TOTAL_SIZE.search(line):
                metrics.total_submission_size_bytes = int(match.group(1))
            if match := RE_ACTIVATION.search(line):
                metrics.mlp_activation = match.group(1)
                metrics.mlp_leaky_slope = float(match.group(2))
                metrics.asqu_beta_init = float(match.group(3))
                metrics.asqu_lr = float(match.group(4))
                metrics.asqu_sigmoid_beta = parse_bool(match.group(5))
            if match := RE_QAT_AND_VE.search(line):
                metrics.late_qat_threshold = float(match.group(1))
                metrics.ve_enabled = parse_bool(match.group(2))
                metrics.ve_layers = match.group(3).strip()
            if match := RE_XSA.search(line):
                metrics.xsa_last_n = int(match.group(1))
                metrics.xsa_active_layers = parse_active_layers(match.group(2))
            if match := RE_ASQU_DIAG.search(line):
                metrics.asqu_diags.append(
                    AsquDiag(
                        layer=int(match.group(1)),
                        beta_mean=float(match.group(2)),
                        beta_std=float(match.group(3)),
                        beta_min=float(match.group(4)),
                        beta_max=float(match.group(5)),
                        frac_near_zero=float(match.group(6)),
                        frac_gt_half=float(match.group(7)),
                    )
                )
            if match := RE_QUANT_DIAG.search(line):
                name = match.group(1)
                metrics.quant_diags[name] = QuantDiag(
                    name=name,
                    mse=float(match.group(2)),
                    shape=match.group(3),
                )
    return metrics


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_delta(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{digits}f}"


def fmt_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def as_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def quant_delta_rows(control: RunMetrics, candidate: RunMetrics) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in sorted(set(control.quant_diags) & set(candidate.quant_diags)):
        c = control.quant_diags[name].mse
        a = candidate.quant_diags[name].mse
        rows.append(
            {
                "name": name,
                "control_mse": c,
                "candidate_mse": a,
                "delta": a - c,
                "ratio": (a / c) if c > 0 else math.inf,
            }
        )
    rows.sort(key=lambda row: row["delta"], reverse=True)
    return rows


def summarize_asqu(run: RunMetrics, beta_init: float = 0.25) -> dict[str, object] | None:
    if not run.asqu_diags:
        return None
    means = [diag.beta_mean for diag in run.asqu_diags]
    stds = [diag.beta_std for diag in run.asqu_diags]
    near_zero = [diag.frac_near_zero for diag in run.asqu_diags]
    gt_half = [diag.frac_gt_half for diag in run.asqu_diags]
    max_abs_shift = max(abs(value - beta_init) for value in means)
    diverged = max_abs_shift >= 0.02 or max(stds) >= 0.02
    return {
        "layer_count": len(run.asqu_diags),
        "mean_of_means": mean(means),
        "min_mean": min(means),
        "max_mean": max(means),
        "max_std": max(stds),
        "mean_frac_near_zero": mean(near_zero),
        "mean_frac_gt_half": mean(gt_half),
        "max_abs_shift_from_init": max_abs_shift,
        "diverged_from_init_heuristic": diverged,
    }


def choose_phase1_decision(control: RunMetrics, candidate: RunMetrics) -> tuple[str, str]:
    control_int6 = control.final_int6_roundtrip_exact_val_bpb
    candidate_int6 = candidate.final_int6_roundtrip_exact_val_bpb
    if control_int6 is not None and candidate_int6 is not None and candidate_int6 < control_int6:
        return "GO to Phase 3", "ASQU beats control on final_int6_roundtrip_exact."
    control_pre, source = control.prequant_metric()
    candidate_pre, _ = candidate.prequant_metric()
    if control_pre is not None and candidate_pre is not None and candidate_pre < control_pre:
        return "GO to Phase 2", f"ASQU is better on {source} but still loses on the quantized endpoint."
    return "KILL", "ASQU is not better on the available pre-quant comparator and still loses after quantization."


def render_phase1(control: RunMetrics, candidate: RunMetrics) -> str:
    control_pre, pre_source = control.prequant_metric()
    candidate_pre, _ = candidate.prequant_metric()
    decision, reason = choose_phase1_decision(control, candidate)
    rows = [
        [
            "End-of-train val_bpb",
            fmt_float(control.end_train_val_bpb),
            fmt_float(candidate.end_train_val_bpb),
            fmt_delta(
                None
                if control.end_train_val_bpb is None or candidate.end_train_val_bpb is None
                else candidate.end_train_val_bpb - control.end_train_val_bpb
            ),
        ],
        [
            f"{pre_source} val_bpb",
            fmt_float(control_pre),
            fmt_float(candidate_pre),
            fmt_delta(None if control_pre is None or candidate_pre is None else candidate_pre - control_pre),
        ],
        [
            "final_int6_roundtrip_exact val_bpb",
            fmt_float(control.final_int6_roundtrip_exact_val_bpb, 8),
            fmt_float(candidate.final_int6_roundtrip_exact_val_bpb, 8),
            fmt_delta(
                None
                if control.final_int6_roundtrip_exact_val_bpb is None
                or candidate.final_int6_roundtrip_exact_val_bpb is None
                else candidate.final_int6_roundtrip_exact_val_bpb - control.final_int6_roundtrip_exact_val_bpb,
                8,
            ),
        ],
        [
            "final_int6_sliding_window_exact val_bpb",
            fmt_float(control.final_int6_sliding_window_exact_val_bpb, 8),
            fmt_float(candidate.final_int6_sliding_window_exact_val_bpb, 8),
            fmt_delta(
                None
                if control.final_int6_sliding_window_exact_val_bpb is None
                or candidate.final_int6_sliding_window_exact_val_bpb is None
                else candidate.final_int6_sliding_window_exact_val_bpb - control.final_int6_sliding_window_exact_val_bpb,
                8,
            ),
        ],
        [
            "Which averaging won",
            control.best_avg_name or "n/a",
            candidate.best_avg_name or "n/a",
            "n/a",
        ],
        [
            "step_avg (ms)",
            fmt_float(control.step_avg_ms, 2),
            fmt_float(candidate.step_avg_ms, 2),
            fmt_delta(
                None if control.step_avg_ms is None or candidate.step_avg_ms is None else candidate.step_avg_ms - control.step_avg_ms,
                2,
            ),
        ],
        [
            "Total submission size (bytes)",
            fmt_int(control.total_submission_size_bytes),
            fmt_int(candidate.total_submission_size_bytes),
            fmt_delta(
                None
                if control.total_submission_size_bytes is None or candidate.total_submission_size_bytes is None
                else float(candidate.total_submission_size_bytes - control.total_submission_size_bytes),
                0,
            ),
        ],
    ]
    lines = [
        "## Phase 1 Analysis",
        "",
        f"Control: `{control.label}`",
        f"ASQU allfix: `{candidate.label}`",
        "",
        as_markdown_table(["Metric", "Control", "ASQU allfix", "Delta"], rows),
        "",
    ]
    quant_rows = quant_delta_rows(control, candidate)
    if quant_rows:
        top_rows = [
            [
                row["name"],
                fmt_float(row["control_mse"], 8),
                fmt_float(row["candidate_mse"], 8),
                fmt_delta(row["delta"], 8),
                "inf" if math.isinf(row["ratio"]) else f"{row['ratio']:.3f}x",
            ]
            for row in quant_rows[:8]
        ]
        lines.extend(
            [
                "### Largest Quant MSE Regressions",
                "",
                as_markdown_table(
                    ["Tensor", "Control MSE", "ASQU MSE", "Delta", "Ratio"],
                    top_rows,
                ),
                "",
            ]
        )
    asqu_summary = summarize_asqu(candidate, beta_init=candidate.asqu_beta_init or 0.25)
    if asqu_summary is not None:
        lines.extend(
            [
                "### ASQU Beta Summary",
                "",
                f"- layers logged: `{asqu_summary['layer_count']}`",
                f"- mean(beta_mean): `{asqu_summary['mean_of_means']:.4f}`",
                f"- beta_mean range: `{asqu_summary['min_mean']:.4f}` to `{asqu_summary['max_mean']:.4f}`",
                f"- max beta_std: `{asqu_summary['max_std']:.4f}`",
                f"- mean frac_near_zero: `{asqu_summary['mean_frac_near_zero']:.4f}`",
                f"- mean frac_gt_half: `{asqu_summary['mean_frac_gt_half']:.4f}`",
                f"- heuristic divergence from init `{candidate.asqu_beta_init or 0.25:.2f}`: "
                f"`{asqu_summary['diverged_from_init_heuristic']}`",
                "",
            ]
        )
    lines.extend([f"Decision: `{decision}`", reason])
    return "\n".join(lines)


def render_phase2(control: RunMetrics, runs: list[RunMetrics]) -> str:
    rows: list[list[str]] = []
    control_int6 = control.final_int6_roundtrip_exact_val_bpb
    rows.append(
        [
            control.label,
            "n/a",
            fmt_float(control.late_qat_threshold, 2),
            fmt_float(control.final_int6_roundtrip_exact_val_bpb, 8),
            "baseline",
        ]
    )
    winner: RunMetrics | None = None
    for run in runs:
        delta = None
        if control_int6 is not None and run.final_int6_roundtrip_exact_val_bpb is not None:
            delta = run.final_int6_roundtrip_exact_val_bpb - control_int6
        if delta is not None and delta < 0 and (
            winner is None
            or (
                winner.final_int6_roundtrip_exact_val_bpb is not None
                and run.final_int6_roundtrip_exact_val_bpb is not None
                and run.final_int6_roundtrip_exact_val_bpb < winner.final_int6_roundtrip_exact_val_bpb
            )
        ):
            winner = run
        rows.append(
            [
                run.label,
                "n/a" if run.asqu_sigmoid_beta is None else str(int(run.asqu_sigmoid_beta)),
                fmt_float(run.late_qat_threshold, 2),
                fmt_float(run.final_int6_roundtrip_exact_val_bpb, 8),
                fmt_delta(delta, 8),
            ]
        )
    decision = (
        f"GO to Phase 3 with `{winner.label}`." if winner is not None else "KILL. No ASQU config beat control on final_int6_roundtrip_exact."
    )
    return "\n".join(
        [
            "## Phase 2 Analysis",
            "",
            as_markdown_table(
                ["Run", "sigmoid_beta", "QAT threshold", "int6_roundtrip_exact", "delta vs control"],
                rows,
            ),
            "",
            f"Decision: {decision}",
        ]
    )


def paired_seed_rows(control_runs: list[RunMetrics], asqu_runs: list[RunMetrics]) -> list[tuple[int, RunMetrics, RunMetrics]]:
    control_by_seed = {run.seed: run for run in control_runs if run.seed is not None}
    asqu_by_seed = {run.seed: run for run in asqu_runs if run.seed is not None}
    overlap = sorted(set(control_by_seed) & set(asqu_by_seed))
    if not overlap:
        raise ValueError("No overlapping seeds were found between control and ASQU logs.")
    return [(seed, control_by_seed[seed], asqu_by_seed[seed]) for seed in overlap]


def render_target_command(run: RunMetrics) -> str:
    env_lines = [
        "OMP_NUM_THREADS=1 \\",
        "PYTHONUNBUFFERED=1 \\",
        "RUN_ID=diag_asqu_winner_seed1337 \\",
        "SEED=1337 \\",
        f"DATA_PATH={DEFAULT_DATA_PATH} \\",
        f"TOKENIZER_PATH={DEFAULT_TOKENIZER_PATH} \\",
        "VOCAB_SIZE=1024 \\",
        f"MLP_ACTIVATION={run.mlp_activation or 'asqu'} \\",
    ]
    if run.asqu_beta_init is not None:
        env_lines.append(f"ASQU_BETA_INIT={run.asqu_beta_init} \\")
    if run.asqu_lr is not None:
        env_lines.append(f"ASQU_LR={run.asqu_lr} \\")
    if run.asqu_sigmoid_beta is not None:
        env_lines.append(f"ASQU_SIGMOID_BETA={int(run.asqu_sigmoid_beta)} \\")
    if run.late_qat_threshold is not None:
        env_lines.append(f"LATE_QAT_THRESHOLD={run.late_qat_threshold} \\")
    if run.xsa_last_n is not None:
        env_lines.append(f"XSA_LAST_N={run.xsa_last_n} \\")
    if run.ve_layers is not None:
        env_lines.append(f"VE_LAYERS={run.ve_layers} \\")
    env_lines.append("MAX_WALLCLOCK_SECONDS=600 \\")
    env_lines.append("torchrun --standalone --nproc_per_node=8 train_gpt.py")
    return "```bash\ncd " + DEFAULT_FOLDER + "\n\n" + "\n".join(env_lines) + "\n```"


def write_or_print(text: str, output: Path | None, write: bool) -> None:
    if output is not None and write:
        output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        return
    print(text)


def render_pr_positive(control_runs: list[RunMetrics], asqu_runs: list[RunMetrics], fixes: list[str]) -> str:
    pairs = paired_seed_rows(control_runs, asqu_runs)
    rows: list[list[str]] = []
    deltas: list[float] = []
    beta_summaries: list[dict[str, object]] = []
    best_candidate: RunMetrics | None = None
    for seed, control, asqu in pairs:
        if control.final_int6_roundtrip_exact_val_bpb is None or asqu.final_int6_roundtrip_exact_val_bpb is None:
            raise ValueError(f"Missing final_int6_roundtrip_exact metric for seed {seed}.")
        delta = asqu.final_int6_roundtrip_exact_val_bpb - control.final_int6_roundtrip_exact_val_bpb
        deltas.append(delta)
        rows.append(
            [
                str(seed),
                fmt_float(control.final_int6_roundtrip_exact_val_bpb, 8),
                fmt_float(asqu.final_int6_roundtrip_exact_val_bpb, 8),
                fmt_delta(delta, 8),
            ]
        )
        summary = summarize_asqu(asqu, beta_init=asqu.asqu_beta_init or 0.25)
        if summary is not None:
            beta_summaries.append(summary)
        if best_candidate is None or (
            best_candidate.final_int6_roundtrip_exact_val_bpb is not None
            and asqu.final_int6_roundtrip_exact_val_bpb < best_candidate.final_int6_roundtrip_exact_val_bpb
        ):
            best_candidate = asqu
    mean_delta = mean(deltas)
    if mean_delta >= 0:
        raise ValueError("Positive PR render requested, but the mean ASQU delta is not better than control.")
    fixes_text = ", ".join(fixes) if fixes else "the updated ASQU recovery stack"
    beta_lines: list[str]
    if beta_summaries:
        mean_of_means = mean([float(summary["mean_of_means"]) for summary in beta_summaries])
        min_mean = min(float(summary["min_mean"]) for summary in beta_summaries)
        max_mean = max(float(summary["max_mean"]) for summary in beta_summaries)
        max_std = max(float(summary["max_std"]) for summary in beta_summaries)
        diverged = any(bool(summary["diverged_from_init_heuristic"]) for summary in beta_summaries)
        beta_lines = [
            f"Across the measured ASQU runs, layerwise beta means ranged from `{min_mean:.4f}` to `{max_mean:.4f}`",
            f"with mean(beta_mean) `{mean_of_means:.4f}` and max beta std `{max_std:.4f}`.",
            (
                "The beta diagnostics do diverge from the `0.25` initialization, so the learned per-channel slope is doing real work."
                if diverged
                else "The beta diagnostics stay close to the `0.25` initialization, so the learned per-channel slope is not moving much. That should be acknowledged explicitly."
            ),
        ]
    else:
        beta_lines = [
            "No `asqu_diag` lines were present in the supplied logs, so beta divergence cannot be claimed yet."
        ]
    if best_candidate is None:
        raise ValueError("No candidate run was available to render the target command.")
    return "\n".join(
        [
            "## Summary",
            "",
            f"ASQU improves the decisive int6 roundtrip metric by `{abs(mean_delta):.8f}` BPB on the March 22 control line in matched 1GPU/2000-step debug runs ({len(pairs)}-seed mean).",
            "",
            "## What Fixed The Regression",
            "",
            f"The original local run showed a `+0.0019 BPB` quantization regression. In the updated stack, that regression is resolved by {fixes_text}.",
            "",
            "## Cross-Seed Consistency",
            "",
            as_markdown_table(
                ["Seed", "Control int6_roundtrip_exact", "ASQU int6_roundtrip_exact", "Delta (ASQU - control)"],
                rows,
            ),
            "",
            f"Mean delta: `{mean_delta:.8f}` BPB",
            f"Std delta: `{stddev(deltas):.8f}` BPB",
            "",
            "## Beta Divergence",
            "",
            *beta_lines,
            "",
            "## Compute Request",
            "",
            "Request a single `8xH100` A/B run first, roughly `$40-60`, to confirm that the local signal transfers to the real `600s` regime.",
            "Only request more compute if that first target-hardware result stays positive on the decisive quantized metric.",
            "",
            "## Winning 8xH100 Command",
            "",
            render_target_command(best_candidate),
        ]
    )


def render_pr_negative(control: RunMetrics, asqu_runs: list[RunMetrics]) -> str:
    if control.final_int6_roundtrip_exact_val_bpb is None:
        raise ValueError("Control log is missing final_int6_roundtrip_exact.")
    rows: list[list[str]] = []
    aggregated_quant: dict[str, list[float]] = {}
    for run in asqu_runs:
        prequant, pre_source = run.prequant_metric()
        delta = None
        if run.final_int6_roundtrip_exact_val_bpb is not None:
            delta = run.final_int6_roundtrip_exact_val_bpb - control.final_int6_roundtrip_exact_val_bpb
        rows.append(
            [
                run.label,
                "n/a" if run.asqu_sigmoid_beta is None else str(int(run.asqu_sigmoid_beta)),
                fmt_float(run.late_qat_threshold, 2),
                fmt_float(prequant),
                pre_source,
                fmt_float(run.final_int6_roundtrip_exact_val_bpb, 8),
                fmt_delta(delta, 8),
            ]
        )
        for quant_row in quant_delta_rows(control, run):
            aggregated_quant.setdefault(str(quant_row["name"]), []).append(float(quant_row["delta"]))
    quant_rows = sorted(
        (
            {
                "name": name,
                "avg_delta": mean(deltas),
            }
            for name, deltas in aggregated_quant.items()
        ),
        key=lambda row: row["avg_delta"],
        reverse=True,
    )
    quant_table = as_markdown_table(
        ["Tensor", "Average MSE delta vs control"],
        [[row["name"], fmt_delta(row["avg_delta"], 8)] for row in quant_rows[:8]],
    ) if quant_rows else "No overlapping `quant_diag` lines were present in the supplied logs."
    return "\n".join(
        [
            "## Summary",
            "",
            "ASQU does not survive int6 quantization in the March 22 stack. Pre-quantized improvements are erased or reversed by the per-row int6 export path.",
            "",
            "## What Was Tried",
            "",
            as_markdown_table(
                ["Run", "sigmoid_beta", "QAT threshold", "Pre-quant val_bpb", "Pre-quant source", "int6_roundtrip_exact", "Delta vs control"],
                rows,
            ),
            "",
            "## Quantization Diagnostics",
            "",
            quant_table,
            "",
            "## Conclusion",
            "",
            "Per-channel learned activation slopes in ASQU appear to create weight distributions that are harder for uniform int6 per-row quantization in this stack.",
            "That closes the ASQU activation question for the current export pipeline unless the quantization method itself changes.",
            "",
            "## Recommendation",
            "",
            "Either move to a more quantization-aware design such as stronger clipping calibration or full GPTQ, or stay with the simpler `LeakyReLU(0.5)^2` / fixed-slope family for this stack.",
        ]
    )


def update_submission(log: RunMetrics, submission_path: Path, write: bool) -> str:
    if log.final_int6_sliding_window_exact_val_bpb is None or log.total_submission_size_bytes is None:
        raise ValueError("The supplied log is missing final_int6_sliding_window_exact or submission size.")
    data = json.loads(submission_path.read_text(encoding="utf-8"))
    data["val_loss"] = None
    data["val_bpb"] = log.final_int6_sliding_window_exact_val_bpb
    data["bytes_total"] = log.total_submission_size_bytes
    rendered = json.dumps(data, indent=2) + "\n"
    if write:
        submission_path.write_text(rendered, encoding="utf-8")
    return rendered


def serialize_run(metrics: RunMetrics) -> dict[str, object]:
    payload = asdict(metrics)
    payload["quant_diags"] = {name: asdict(diag) for name, diag in metrics.quant_diags.items()}
    payload["prequant_metric"] = {
        "value": metrics.prequant_metric()[0],
        "source": metrics.prequant_metric()[1],
    }
    summary = summarize_asqu(metrics, beta_init=metrics.asqu_beta_init or 0.25)
    if summary is not None:
        payload["asqu_summary"] = summary
    return payload


def parse_named_log(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label, Path(path)
    path = Path(raw)
    return path.stem, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automation for TASKS.md analysis and safe artifact updates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse", help="Parse one or more logs and print JSON.")
    parse_cmd.add_argument("logs", nargs="+", help="Log paths or LABEL=PATH.")

    phase1_cmd = subparsers.add_parser("phase1", help="Render the TASK-1.3 comparison.")
    phase1_cmd.add_argument("--control", required=True, help="Control log path.")
    phase1_cmd.add_argument("--candidate", required=True, help="ASQU allfix log path.")

    phase2_cmd = subparsers.add_parser("phase2", help="Render the TASK-2.4 comparison.")
    phase2_cmd.add_argument("--control", required=True, help="Control log path.")
    phase2_cmd.add_argument(
        "--run",
        action="append",
        default=[],
        help="Candidate log in LABEL=PATH form. Repeat for each ASQU config.",
    )

    positive_cmd = subparsers.add_parser("render-pr-positive", help="Render TASK-3.4 from measured logs.")
    positive_cmd.add_argument("--control-log", action="append", required=True, help="Control log path. Repeat for each seed.")
    positive_cmd.add_argument("--asqu-log", action="append", required=True, help="ASQU winning log path. Repeat for each seed.")
    positive_cmd.add_argument("--fix", action="append", default=[], help="Named fix to mention in the PR.")
    positive_cmd.add_argument("--output", type=Path, help="Optional PR_DRAFT.md path.")
    positive_cmd.add_argument("--write", action="store_true", help="Write to --output instead of printing.")

    negative_cmd = subparsers.add_parser("render-pr-negative", help="Render TASK-4.1 from measured logs.")
    negative_cmd.add_argument("--control-log", required=True, help="Control log path.")
    negative_cmd.add_argument(
        "--asqu-log",
        action="append",
        required=True,
        help="ASQU log in LABEL=PATH form. Repeat for each failed config.",
    )
    negative_cmd.add_argument("--output", type=Path, help="Optional PR_DRAFT.md path.")
    negative_cmd.add_argument("--write", action="store_true", help="Write to --output instead of printing.")

    submission_cmd = subparsers.add_parser("update-submission", help="Render or update submission.json from a measured log.")
    submission_cmd.add_argument("--log", required=True, help="Winning seed log path.")
    submission_cmd.add_argument("--submission", required=True, type=Path, help="submission.json path.")
    submission_cmd.add_argument("--write", action="store_true", help="Write the updated JSON instead of printing it.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "parse":
        payload = []
        for raw in args.logs:
            label, path = parse_named_log(raw)
            payload.append(serialize_run(parse_log(path, label=label)))
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "phase1":
        control = parse_log(Path(args.control), label="control")
        candidate = parse_log(Path(args.candidate), label="asqu_allfix")
        print(render_phase1(control, candidate))
        return 0

    if args.command == "phase2":
        control = parse_log(Path(args.control), label="control")
        runs = [parse_log(path, label=label) for label, path in (parse_named_log(raw) for raw in args.run)]
        if not runs:
            raise ValueError("At least one --run LABEL=PATH candidate is required.")
        print(render_phase2(control, runs))
        return 0

    if args.command == "render-pr-positive":
        control_runs = [parse_log(Path(path)) for path in args.control_log]
        asqu_runs = [parse_log(Path(path)) for path in args.asqu_log]
        rendered = render_pr_positive(control_runs, asqu_runs, fixes=args.fix)
        write_or_print(rendered, args.output, args.write)
        return 0

    if args.command == "render-pr-negative":
        control = parse_log(Path(args.control_log), label="control")
        asqu_runs = [parse_log(path, label=label) for label, path in (parse_named_log(raw) for raw in args.asqu_log)]
        rendered = render_pr_negative(control, asqu_runs)
        write_or_print(rendered, args.output, args.write)
        return 0

    if args.command == "update-submission":
        rendered = update_submission(parse_log(Path(args.log), label="winner"), args.submission, write=args.write)
        if not args.write:
            print(rendered)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
