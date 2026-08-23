## Installation

### Prerequisites

* Python 3.13 (the project pins `>=3.13,<3.14`).
* [uv](https://docs.astral.sh/uv/) for dependency management.
* FSL BET available on `PATH`.
* CUDA is optional/

### Clone the repository

```powershell
git clone https://github.com/v-sundaresan/microbleed-detection.git
cd microbleed-detection
```

### Install dependencies

Sync the locked project environment, including optional dependency groups:

```powershell
uv sync --frozen --all-groups
```

### Verify the CLI

```powershell
uv run microbleednet --help
```

This should print the available commands.