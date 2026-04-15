"""Rich console output for Growt audit reports."""

from __future__ import annotations

from typing import Optional

from growt_client.models import AuditResult, MetricsResult, QuantizationAuditResult

# SQNR quality bands
_SQNR_BANDS = [(20, "EXCELLENT"), (15, "GOOD"), (10, "MODERATE"), (6, "POOR"), (0, "CRITICAL")]


def _sqnr_label(db: float | None) -> str:
    if db is None:
        return "N/A"
    for threshold, label in _SQNR_BANDS:
        if db >= threshold:
            return f"{db:.1f} dB ({label})"
    return f"{db:.1f} dB (CRITICAL)"


def _diagnosis_marker(d: str | None) -> str:
    if d == "SAFE":
        return "SAFE"
    if d == "RED_FLAG":
        return "!! RED_FLAG !!"
    return d or "INCONCLUSIVE"


def format_audit_report(
    result: AuditResult,
    metrics: Optional[MetricsResult] = None,
    title: str = "GROWT TRANSFER AUDIT",
) -> str:
    """Format a single audit result as a console report."""
    w = 62
    lines = [
        "=" * w,
        f"  {title}".center(w),
        "=" * w,
        "",
        f"  Diagnosis:       {_diagnosis_marker(result.diagnosis)}",
        f"  Transfer Oracle: {result.transfer_oracle or 'N/A'}",
        f"  Coverage:        {f'{result.coverage_pct:.1%}' if result.coverage_pct else 'N/A'}",
        f"  Consensus:       {result.consensus_quality or 'N/A'}",
        f"  Flagged Samples: {result.n_flagged_samples}",
    ]

    if metrics:
        lines.append(f"  SQNR:            {_sqnr_label(metrics.sqnr_db)}")
        if metrics.rank_correlation is not None:
            lines.append(f"  Rank Preserv.:   {metrics.rank_correlation:.3f}")

    if result.classes_at_risk:
        lines.append("")
        lines.append("  Classes at Risk:")
        for cls in result.classes_at_risk:
            lines.append(f"    !! {cls}")

    if result.recommendations:
        lines.append("")
        lines.append("  Recommendations:")
        for rec in result.recommendations[:3]:
            lines.append(f"    - {rec}")

    lines.append("")
    lines.append("=" * w)
    return "\n".join(lines)


def format_quantization_report(
    result: QuantizationAuditResult,
    metrics_per_variant: Optional[dict[str, MetricsResult]] = None,
) -> str:
    """Format a multi-variant quantization comparison."""
    w = 62
    lines = [
        "=" * w,
        "  GROWT QUANTIZATION COMPARISON".center(w),
        "=" * w,
        "",
        f"  {'Variant':<12} {'Oracle':>8} {'Coverage':>10} {'SQNR':>12} {'Diagnosis':<16}",
        f"  {'-'*12} {'-'*8} {'-'*10} {'-'*12} {'-'*16}",
    ]

    for v in result.variants:
        sqnr_str = "N/A"
        if metrics_per_variant and v.name in metrics_per_variant:
            m = metrics_per_variant[v.name]
            sqnr_str = _sqnr_label(m.sqnr_db) if m.sqnr_db else "N/A"

        oracle_str = f"{v.ensemble_oracle:.2f}" if v.ensemble_oracle is not None else "N/A"
        coverage_str = f"{v.coverage_pct:.1%}" if v.coverage_pct is not None else "N/A"
        diag_str = _diagnosis_marker(v.diagnosis)

        lines.append(f"  {v.name:<12} {oracle_str:>8} {coverage_str:>10} {sqnr_str:>12} {diag_str:<16}")

    if result.recommended_level:
        lines.append("")
        lines.append(f"  Recommended: {result.recommended_level}")

    lines.append("")
    lines.append("=" * w)
    return "\n".join(lines)


def format_monitor_summary(
    total: int, familiar: int, boundary: int, flagged: int,
    last_diagnosis: str | None = None,
) -> str:
    """One-line monitor summary for Triton logging."""
    pct_familiar = f"{familiar/total:.1%}" if total > 0 else "N/A"
    pct_flagged = f"{flagged/total:.1%}" if total > 0 else "N/A"
    diag = f" | Last audit: {last_diagnosis}" if last_diagnosis else ""
    return (
        f"[Growt Monitor] {total} inferences | "
        f"{pct_familiar} familiar | {pct_flagged} flagged{diag}"
    )


def format_training_trajectory(
    history: list[tuple[int, AuditResult]],
) -> str:
    """Format epoch-over-epoch audit trajectory."""
    if not history:
        return "  No audit history."

    lines = ["  Training Trajectory:"]
    for epoch, result in history:
        oracle = f"{result.transfer_oracle:.2f}" if result.transfer_oracle else "N/A"
        cov = f"{result.coverage_pct:.0%}" if result.coverage_pct else "N/A"
        diag = result.diagnosis or "?"
        marker = " <- final" if (epoch, result) == history[-1] else ""
        lines.append(f"  Epoch {epoch:>3}: {diag:<14} oracle={oracle}  coverage={cov}{marker}")

    return "\n".join(lines)
