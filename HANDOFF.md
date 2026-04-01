# Handoff

This repository is the standalone Python-only `hybrid-mechlab-python` SDK.

- Runtime dependency: `numpy>=2.0`
- Supported math backend: `math_backend="python"`
- Import package: `hybrid_mechlab`
- No Rust companion, BLT sidecar, or monorepo workspace is required

## Short Start

Create a virtual environment, install from source, and run the quickstart:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install .
python - <<'PY'
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
PY
```

Expected output shape:

- one trace summary line
- one integer persistence count

## Engineer Validation

Run the local verification suite:

```bash
python3 -m pytest
python3 -m build
python3 -m twine check dist/*
```

Validate built artifacts in fresh environments:

```bash
tmpdir=$(mktemp -d)
python3 -m venv "$tmpdir/venv"
source "$tmpdir/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install dist/*.whl
python - <<'PY'
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
PY
```

Repeat the same flow against `dist/*.tar.gz` to validate the source distribution.

## Notes

- The repository is intentionally Python-only.
- `math_backend="rust"` is not available in this repo.
- The current supported public surface is:
  - `HybridLab`
  - `TraceHandle`
  - `profiles`
  - `hybrid_mechlab.kernel`
  - `hybrid_mechlab.topology.offline`
  - `hybrid_mechlab.topology.metrics`
  - `hybrid_mechlab.topology.online`
  - `hybrid_mechlab.topology.sheaf`
  - `hybrid_mechlab.experiments.long_context`
