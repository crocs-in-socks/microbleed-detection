# MicrobleedNet

## Documentation

- [Architecture](docs/architecture.md) — layering, data flow, and model overview.

## Setup and installation

### Pre-requisites

Install [uv](https://docs.astral.sh/uv/), which manages the Python version and the project environment:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Preprocessing also requires **FSL BET** on `PATH`. FSL is not natively supported on Windows; use WSL or a Linux/macOS host for the preprocessing step. CUDA is optional, as CPU inference and training are also possible.

### Install the project

```powershell
uv sync --frozen --all-groups
```

The project targets Python 3.13 (`>=3.13,<3.14`), which uv downloads automatically if it is not already present.

To also install the optional plotting dependencies (matplotlib), add the extra:

```powershell
uv sync --frozen --all-groups --extra plots
```
