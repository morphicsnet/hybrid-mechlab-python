# hybrid-mechlab-python

Standalone NumPy-first Python SDK for hybrid transport research.

This repository packages the Python trace, schedule, kernel, and offline
topology surfaces as a self-contained project. It is intentionally Python-only
and does not depend on the Rust workspace, BLT sidecar, or model-runtime
integrations.

## Install

```bash
pip install .
```

- The only runtime dependency is `numpy>=2.0`.
- `math_backend="python"` is the supported backend in this repo.
- The install name is `hybrid-mechlab-python`.
- The import package remains `hybrid_mechlab`.

## Quickstart

```python
from hybrid_mechlab import HybridLab, profiles
from hybrid_mechlab.topology.offline import compute_persistence

lab = HybridLab.attach(
    model="dummy-qwen",
    profile=profiles.reference.qwen35(),
    backend="adapter",
)
trace = lab.run(prompts=["Measure topology for a reference hybrid."])
print(trace.summary())

report = compute_persistence(trace)
print(report.summary.h0_pairs)
```

## Supported Surface

- `HybridLab`
- `TraceHandle`
- `profiles`
- `hybrid_mechlab.kernel`
- `hybrid_mechlab.topology.offline`
- `hybrid_mechlab.topology.metrics`
- `hybrid_mechlab.topology.online`
- `hybrid_mechlab.topology.sheaf`
- `hybrid_mechlab.experiments.long_context`

## Validation

```bash
python3 -m pytest
python3 -m build
python3 -m twine check dist/*
```

See [HANDOFF.md](HANDOFF.md) for zip-ready transfer and validation instructions.
