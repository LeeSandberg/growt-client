# growt-client

**Python client for the [Growt Transfer Oracle API](https://transferoracle.ai)** — structural model auditing before deployment.

[![License: MPL-2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

## What is this?

Shared Python client used by all Growt NVIDIA plugins. Provides typed API methods, result dataclasses, and rich console formatting.

## Install

```bash
pip install growt-client
# With rich console output:
pip install growt-client[rich]
```

## Quick Start

```python
from growt_client import GrowtClient, format_audit_report

client = GrowtClient(api_url="http://your-growt-api:8000", api_key="your-key")

# Audit model transfer
result = client.audit_transfer(
    features_train=train_features,  # [[float]] from training
    labels_train=train_labels,       # [int]
    features_deploy=deploy_features, # [[float]] from deployment
)

print(format_audit_report(result))
# Diagnosis: SAFE | RED_FLAG | BAD_MODEL | UNDERTRAINED
```

## API Methods

| Method | Endpoint | Use Case |
|--------|----------|----------|
| `audit_transfer()` | `POST /v1/audit/transfer` | Full structural transfer audit |
| `audit_quantization()` | `POST /v1/audit/quantization` | Compare quantization variants |
| `metrics_compare()` | `POST /v1/metrics/compare` | SQNR, cosine, rank preservation |
| `monitor_state()` | `POST /v1/monitor/state` | Real-time single-vector check |
| `session_create()` | Session management | Create monitor reference session |

## Result Types

- `AuditResult` — diagnosis, transfer_oracle, coverage_pct, classes_at_risk
- `MetricsResult` — sqnr_db, cosine_mean, rank_correlation
- `MonitorResult` — status (familiar/boundary/flagged), anomaly_score
- `QuantizationAuditResult` — per-variant comparison, recommended_level

## License

[MPL-2.0](LICENSE) — modifications to this code must stay open source.

## Status & Contributing

This is an early release to get the integration started. The code works but is not battle-tested in production yet. We welcome contributions:

- Bug fixes and improvements — PRs welcome
- New features and endpoint integrations
- Better error handling and edge cases
- Documentation improvements
- Test coverage

Open an issue or submit a PR on GitHub. All contributions must be compatible with the MPL-2.0 license.


## Related

- [Documentation](https://transferoracle.ai/growt/docs) — API reference, all plugins, tiers
- [growt-client](https://github.com/LeeSandberg/growt-client) — Python client (shared by all plugins)
- [growt-modelopt](https://github.com/LeeSandberg/growt-modelopt) — NVIDIA ModelOpt
- [growt-quark](https://github.com/LeeSandberg/growt-quark) — AMD Quark
- [growt-nemo](https://github.com/LeeSandberg/growt-nemo) — NeMo / PyTorch Lightning
- [growt-vllm](https://github.com/LeeSandberg/growt-vllm) — vLLM (NVIDIA + AMD)
- [growt-triton](https://github.com/LeeSandberg/growt-triton) — Triton Inference Server
- [growt-trt-validator](https://github.com/LeeSandberg/growt-trt-validator) — TensorRT validator
- [growt-tao](https://github.com/LeeSandberg/growt-tao) — TAO Toolkit
- [mlflow-growt](https://github.com/LeeSandberg/mlflow-growt) — MLflow evaluator + Model Registry gate
- [growt-huggingface](https://github.com/LeeSandberg/growt-huggingface) — HuggingFace TrainerCallback + Model Card
- [growt-wandb](https://github.com/LeeSandberg/growt-wandb) — W&B callback, artifact metadata, registry gate

