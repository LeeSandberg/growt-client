"""Result types for all Growt API endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class AuditResult:
    """Result from POST /v1/audit/transfer."""

    safe_to_deploy: bool | None
    diagnosis: str  # SAFE | RED_FLAG | BAD_MODEL | UNDERTRAINED | INCONCLUSIVE
    transfer_oracle: float | None = None
    coverage_pct: float | None = None
    consensus_quality: str | None = None
    n_flagged_samples: int = 0
    classes_at_risk: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    report: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantResult:
    """Per-variant result within a quantization audit."""

    name: str
    diagnosis: str | None = None
    ensemble_oracle: float | None = None
    coverage_pct: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantizationAuditResult:
    """Result from POST /v1/audit/quantization."""

    recommended_level: str | None
    variants: list[VariantResult] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricsResult:
    """Result from POST /v1/metrics/compare."""

    cosine_mean: float | None = None
    cosine_std: float | None = None
    sqnr_db: float | None = None
    noise_ratio: float | None = None
    rank_correlation: float | None = None
    perfect_rank_pct: float | None = None
    summary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MonitorResult:
    """Result from POST /v1/monitor/state."""

    status: str  # familiar | boundary | flagged
    anomaly_score: float = 0.0
    node_id: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def flagged(self) -> bool:
        return self.status == "flagged"


@dataclass(frozen=True)
class BufferStatus:
    """Result from GET /v1/monitor/buffer/status."""

    size: int = 0
    coherence_score: float | None = None
    trigger_ready: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionInfo:
    """Result from session creation."""

    session_id: str
    raw: dict[str, Any] = field(default_factory=dict)
