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

## All Growt Plugins

Open-source plugins and SDKs for the [Transfer Oracle](https://transferoracle.ai) structural AI auditing API.
Plugin code is MPL-2.0; API access is commercial and requires an [API key](https://transferoracle.ai/growt/plugins).

| Plugin | Platform | What it does |
|--------|----------|-------------|
| [growt-client](https://github.com/LeeSandberg/growt-client) | Core | Python client library |
| [growt-modelopt](https://github.com/LeeSandberg/growt-modelopt) | NVIDIA | ModelOpt quantization audit |
| [growt-quark](https://github.com/LeeSandberg/growt-quark) | AMD | Quark quantization audit |
| [growt-nemo](https://github.com/LeeSandberg/growt-nemo) | NVIDIA | NeMo / PyTorch Lightning callback |
| [growt-vllm](https://github.com/LeeSandberg/growt-vllm) | NVIDIA + AMD | vLLM inference monitor |
| [growt-triton](https://github.com/LeeSandberg/growt-triton) | NVIDIA | Triton Inference Server monitor |
| [growt-trt-validator](https://github.com/LeeSandberg/growt-trt-validator) | NVIDIA | TensorRT engine validator |
| [growt-tao](https://github.com/LeeSandberg/growt-tao) | NVIDIA | TAO Toolkit pipeline |
| [mlflow-growt](https://github.com/LeeSandberg/mlflow-growt) | MLflow | Evaluator + deployment gate |
| [growt-huggingface](https://github.com/LeeSandberg/growt-huggingface) | HuggingFace | TrainerCallback + Model Card |
| [growt-wandb](https://github.com/LeeSandberg/growt-wandb) | W&B | Callback + artifact + registry gate |
| [growt-airflow](https://github.com/LeeSandberg/growt-airflow) | Airflow | Pre-deployment audit operator |
| [growt-kubeflow](https://github.com/LeeSandberg/growt-kubeflow) | Kubeflow | Pipeline validation component |
| [growt-kserve](https://github.com/LeeSandberg/growt-kserve) | KServe | Pre-serve validation transformer |
| [growt-dagster](https://github.com/LeeSandberg/growt-dagster) | Dagster | Asset + resource for audit |
| [growt-dvc](https://github.com/LeeSandberg/growt-dvc) | DVC | Pre-push model validation |
| [growt-bentoml](https://github.com/LeeSandberg/growt-bentoml) | BentoML | Pre-serve validation hook |
| [growt-argo](https://github.com/LeeSandberg/growt-argo) | Argo | Workflow validation template |
| [growt-prefect](https://github.com/LeeSandberg/growt-prefect) | Prefect | Task + block for audit |
| [growt-clearml](https://github.com/LeeSandberg/growt-clearml) | ClearML | Pipeline step + callback |
| [growt-docker](https://github.com/LeeSandberg/growt-docker) | Docker/OCI | Audit metadata in containers |

**Links:** [API Docs](https://transferoracle.ai/growt/docs) · [Get API Key](https://transferoracle.ai/growt/plugins) · [LLM Benchmark](https://transferoracle.ai/growt/llm-benchmark) · [All Benchmarks](https://transferoracle.ai/benchmarks)
