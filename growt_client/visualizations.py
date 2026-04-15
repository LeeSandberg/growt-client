"""Matplotlib visualizations for Growt audit results.

Used by NeMo (TensorBoard/WandB), TAO (TensorBoard), and standalone.
Optional dependency — only imported when visualization is requested.
"""

from __future__ import annotations

from typing import Optional

from growt_client.models import AuditResult, MetricsResult


def plot_per_class_coverage(
    audit: AuditResult,
    title: str = "Growt Per-Class Coverage",
) -> "matplotlib.figure.Figure":
    """Bar chart of per-class coverage with at-risk classes highlighted."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    per_class = audit.raw.get("per_class", audit.raw.get("coverage", {}).get("per_class", {}))

    if not per_class or not isinstance(per_class, dict):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No per-class data available", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        return fig

    classes = list(per_class.keys())
    coverages = [per_class[c].get("coverage_pct", 0) if isinstance(per_class[c], dict) else per_class[c] for c in classes]
    at_risk = set(audit.classes_at_risk)

    colors = ["#ef4444" if c in at_risk else "#22d3ee" for c in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.8), 5))
    bars = ax.bar(classes, coverages, color=colors, edgecolor="#1e293b", linewidth=0.5)
    ax.set_ylabel("Coverage %")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.8, color="#f59e0b", linestyle="--", alpha=0.5, label="80% threshold")
    ax.legend()

    for bar, cov in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{cov:.0%}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_training_trajectory(
    history: list[tuple[int, AuditResult]],
    metrics_history: Optional[list[tuple[int, MetricsResult]]] = None,
    title: str = "Growt Training Trajectory",
) -> "matplotlib.figure.Figure":
    """Line chart of transfer oracle + coverage over training epochs."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    epochs = [e for e, _ in history]
    oracles = [r.transfer_oracle or 0 for _, r in history]
    coverages = [r.coverage_pct or 0 for _, r in history]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_oracle = "#22d3ee"
    color_coverage = "#a78bfa"

    ax1.plot(epochs, oracles, "o-", color=color_oracle, label="Transfer Oracle", linewidth=2)
    ax1.plot(epochs, coverages, "s--", color=color_coverage, label="Coverage", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)

    if metrics_history:
        ax2 = ax1.twinx()
        sqnr_epochs = [e for e, _ in metrics_history]
        sqnrs = [m.sqnr_db or 0 for _, m in metrics_history]
        ax2.plot(sqnr_epochs, sqnrs, "^:", color="#f59e0b", label="SQNR (dB)", linewidth=1.5)
        ax2.set_ylabel("SQNR (dB)")
        ax2.legend(loc="upper left")

    # Color background by diagnosis
    for i, (epoch, result) in enumerate(history):
        if result.diagnosis == "SAFE":
            ax1.axvspan(epoch - 0.5, epoch + 0.5, alpha=0.05, color="green")
        elif result.diagnosis == "RED_FLAG":
            ax1.axvspan(epoch - 0.5, epoch + 0.5, alpha=0.1, color="red")

    plt.tight_layout()
    return fig


def plot_quantization_comparison(
    variant_names: list[str],
    sqnr_values: list[float],
    coverage_values: list[float],
    title: str = "Growt Quantization Comparison",
) -> "matplotlib.figure.Figure":
    """Grouped bar chart comparing quantization variants."""
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")

    x = np.arange(len(variant_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(8, len(variant_names) * 2), 5))

    bars1 = ax1.bar(x - width / 2, coverage_values, width, label="Coverage", color="#22d3ee", edgecolor="#1e293b")
    ax1.set_ylabel("Coverage")
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, sqnr_values, width, label="SQNR (dB)", color="#f59e0b", edgecolor="#1e293b")
    ax2.set_ylabel("SQNR (dB)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(variant_names)
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    return fig
