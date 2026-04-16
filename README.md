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

