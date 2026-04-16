"""Growt Transfer Oracle API client — shared across all NVIDIA plugins."""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests

from growt_client.models import (
    AuditResult,
    BufferStatus,
    MetricsResult,
    MonitorResult,
    QuantizationAuditResult,
    SessionInfo,
    VariantResult,
)

logger = logging.getLogger("growt_client")

DEFAULT_TIMEOUT = 120


class GrowtClient:
    """HTTP client for the Growt Transfer Oracle API.

    Args:
        api_url: Base URL (e.g. ``https://api.transferoracle.ai``).
        api_key: API key sent as ``X-API-Key`` header.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_url: str = "https://api.transferoracle.ai",
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["X-API-Key"] = api_key
        self._session.headers["Content-Type"] = "application/json"

    # ------------------------------------------------------------------
    # Core audit
    # ------------------------------------------------------------------

    def audit_transfer(
        self,
        features_train: list[list[float]],
        labels_train: list[int | str],
        features_deploy: list[list[float]],
        labels_deploy: Optional[list[int | str]] = None,
        val_accuracy: Optional[float] = None,
        n_components: int = 20,
        top_k: int = 5,
    ) -> AuditResult:
        """POST /v1/audit/transfer — full structural transfer audit."""
        payload: dict[str, Any] = {
            "features_train": features_train,
            "labels_train": labels_train,
            "features_deploy": features_deploy,
            "n_components": n_components,
            "top_k": top_k,
        }
        if labels_deploy is not None:
            payload["labels_deploy"] = labels_deploy
        if val_accuracy is not None:
            payload["val_accuracy"] = val_accuracy

        data = self._post("/v1/audit/transfer", payload)
        return self._parse_audit(data)

    # ------------------------------------------------------------------
    # Quantization audit
    # ------------------------------------------------------------------

    def audit_quantization(
        self,
        features_reference: list[list[float]],
        labels: list[int | str],
        variants: dict[str, list[list[float]]],
    ) -> QuantizationAuditResult:
        """POST /v1/audit/quantization — compare N quantization variants.

        ``variants`` is a dict mapping variant names to feature lists::

            {"FP8": [[...]], "INT8": [[...]]}
        """
        payload = {
            "features_train": features_reference,
            "labels_train": labels,
            "variants": variants,
        }
        data = self._post("/v1/audit/quantization", payload)

        variant_results = []
        for v in data.get("variants", []):
            if isinstance(v, dict):
                variant_results.append(VariantResult(
                    name=v.get("name", "unknown"),
                    diagnosis=v.get("diagnosis"),
                    ensemble_oracle=v.get("ensemble_oracle"),
                    coverage_pct=v.get("coverage", {}).get("coverage_pct") if isinstance(v.get("coverage"), dict) else v.get("coverage_pct"),
                    raw=v,
                ))

        return QuantizationAuditResult(
            recommended_level=data.get("recommended_level"),
            variants=variant_results,
            raw=data,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics_compare(
        self,
        features_reference: list[list[float]],
        features_compare: list[list[float]],
        labels_reference: Optional[list[int | str]] = None,
    ) -> MetricsResult:
        """POST /v1/metrics/compare — cosine, SQNR, rank preservation.

        Note: ``features_reference`` and ``features_compare`` must have the
        same number of samples (paired comparison).
        """
        payload: dict[str, Any] = {
            "features_reference": features_reference,
            "features_compare": features_compare,
        }
        if labels_reference is not None:
            payload["labels_reference"] = labels_reference

        data = self._post("/v1/metrics/compare", payload)

        cosine = data.get("cosine", {})
        sqnr = data.get("sqnr", {})
        rank = data.get("rank_preservation", {})

        return MetricsResult(
            cosine_mean=cosine.get("mean") if isinstance(cosine, dict) else None,
            cosine_std=cosine.get("std") if isinstance(cosine, dict) else None,
            sqnr_db=sqnr.get("sqnr_db") if isinstance(sqnr, dict) else None,
            noise_ratio=sqnr.get("noise_ratio") if isinstance(sqnr, dict) else None,
            rank_correlation=rank.get("rank_correlation_mean") if isinstance(rank, dict) else None,
            perfect_rank_pct=rank.get("perfect_rank_pct") if isinstance(rank, dict) else None,
            summary=data.get("summary", ""),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Editability & Recovery (quantization-specific)
    # ------------------------------------------------------------------

    def editability_predict(
        self,
        features_original: list[list[float]],
        labels: list[int | str],
        features_compressed: list[list[float]],
    ) -> dict:
        """POST /v1/editability/predict — which classes are recoverable.

        ``features_original`` and ``features_compressed`` must have the same
        number of samples (paired, same order).

        Returns per_class editability scores + ranking.
        """
        return self._post("/v1/editability/predict", {
            "features_original": features_original,
            "labels": labels,
            "features_compressed": features_compressed,
        })

    def audit_safe_level(
        self,
        features_train: list[list[float]],
        labels_train: list[int | str],
        variants: dict[str, list[list[float]]],
    ) -> dict:
        """POST /v1/audit/safe-level — auto-pick safest quantization level.

        Returns recommended_level, all_levels with per-variant safety, rationale.
        """
        return self._post("/v1/audit/safe-level", {
            "features_train": features_train,
            "labels_train": labels_train,
            "variants": variants,
        })

    def audit_lora_recovery(
        self,
        features_train: list[list[float]],
        labels_train: list[int | str],
        features_fp32: list[list[float]],
        features_quantized: list[list[float]],
        features_lora: list[list[float]],
    ) -> dict:
        """POST /v1/audit/lora-recovery — three-way comparison.

        Verdict: NO_DAMAGE / RECOVERABLE / PARTIALLY_RECOVERABLE / DESTRUCTIVE
        """
        return self._post("/v1/audit/lora-recovery", {
            "features_train": features_train,
            "labels_train": labels_train,
            "features_fp32": features_fp32,
            "features_quantized": features_quantized,
            "features_lora": features_lora,
        })

    # ------------------------------------------------------------------
    # Novelty Detection
    # ------------------------------------------------------------------

    def novelty_check(
        self,
        features_ref: list[list[float]],
        features_new: list[list[float]],
    ) -> dict:
        """POST /v1/novelty/check — are new vectors familiar to reference?

        Label-free. Returns scores, flagged_indices, stats.
        """
        return self._post("/v1/novelty/check", {
            "features_ref": features_ref,
            "features_new": features_new,
        })

    def novelty_drift(
        self,
        features_baseline: list[list[float]],
        labels_baseline: list[int | str],
        features_current: list[list[float]],
        labels_current: list[int | str],
    ) -> dict:
        """POST /v1/novelty/drift — per-class centroid shift + spread change.

        Returns per_class drift scores, overall_drift, alerts.
        """
        return self._post("/v1/novelty/drift", {
            "features_baseline": features_baseline,
            "labels_baseline": labels_baseline,
            "features_current": features_current,
            "labels_current": labels_current,
        })

    # ------------------------------------------------------------------
    # Monitor (real-time, sub-ms)
    # ------------------------------------------------------------------

    def monitor_state(self, session_id: str, vector: list[float]) -> MonitorResult:
        """POST /v1/monitor/state — single-vector familiarity check."""
        data = self._post("/v1/monitor/state", {
            "session_id": session_id,
            "vector": vector,
        })
        return MonitorResult(
            status=data.get("status", "unknown"),
            anomaly_score=data.get("anomaly_score", 0.0),
            node_id=data.get("node_id"),
            raw=data,
        )

    def monitor_buffer_append(self, session_id: str, vectors: list[list[float]]) -> dict:
        """POST /v1/monitor/buffer/append — buffer flagged vectors."""
        return self._post("/v1/monitor/buffer/append", {
            "session_id": session_id,
            "vectors": vectors,
        })

    def monitor_buffer_audit(self, session_id: str) -> AuditResult:
        """POST /v1/monitor/buffer/audit — batch re-audit buffered vectors."""
        data = self._post("/v1/monitor/buffer/audit", {"session_id": session_id})
        return self._parse_audit(data)

    def monitor_buffer_status(self, session_id: str) -> BufferStatus:
        """GET /v1/monitor/buffer/status."""
        data = self._get(f"/v1/monitor/buffer/status?session_id={session_id}")
        return BufferStatus(
            size=data.get("size", 0),
            coherence_score=data.get("coherence_score"),
            trigger_ready=data.get("trigger_ready", False),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def session_create(
        self,
        reference_features: list[list[float]],
        reference_labels: list[int | str],
        name: str = "",
    ) -> SessionInfo:
        """Create a monitor session with reference data."""
        data = self._post("/v1/session/create", {
            "features": reference_features,
            "labels": reference_labels,
            "name": name,
        })
        return SessionInfo(
            session_id=data.get("session_id", ""),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.api_url}{path}"
        logger.debug("POST %s (%d bytes)", url, len(str(payload)))
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> dict:
        url = f"{self.api_url}{path}"
        logger.debug("GET %s", url)
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_audit(data: dict) -> AuditResult:
        coverage = data.get("coverage", {})
        consensus = data.get("consensus", {})
        return AuditResult(
            safe_to_deploy=data.get("safe_to_deploy"),
            diagnosis=data.get("diagnosis") or "INCONCLUSIVE",
            transfer_oracle=data.get("ensemble_oracle"),
            coverage_pct=coverage.get("coverage_pct") if isinstance(coverage, dict) else data.get("coverage_pct"),
            consensus_quality=consensus.get("pct_high") if isinstance(consensus, dict) else None,
            n_flagged_samples=len(consensus.get("flagged_indices", [])) if isinstance(consensus, dict) else 0,
            classes_at_risk=data.get("classes_at_risk", []),
            recommendations=data.get("top_recommendations", data.get("recommendations", [])),
            report=data.get("report", ""),
            raw=data,
        )
