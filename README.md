#  🧩 RoboML Research Template

A lightweight, opinionated template for running machine learning and robotics research projects. It provides a standardized structure for experiments, data handling, logging, and evaluation, with ready-to-use configs and scripts for reproducible workflows.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-purple.svg)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)



## Projects Built with RoboML Research Template

Projects that used RoboML template:

[![proteinRL](https://img.shields.io/badge/lorenzomagnino-proteinRL-4c1d95?style=flat-square&logo=github&logoColor=white)](https://github.com/lorenzomagnino/proteinRL)
[![dedaFP](https://img.shields.io/badge/lorenzomagnino-dedaFP-4c1d95?style=flat-square&logo=github&logoColor=white)](https://github.com/lorenzomagnino/dedaFP)

Let me know if you use this template!!


## Features

- **Hydra Configuration Management**: Modular config structure with separate files for model, training, and wandb settings
- **Type-Safe Configs**: Dataclass schemas for configuration validation and IDE support
- **Wandb Integration**: Built-in experiment tracking and logging
- **Fast Package Management**: Uses `uv` for quick dependency installation
- **Development Tools**: Pre-commit hooks with ruff and black for code quality
- **Auto-formatting**: Ruff automatically formats Python files on save (VS Code/Cursor)

## Setup

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd roboml-research-template
```

2. Install dependencies:
```bash
uv venv
source .venv/bin/activate
make install
```

**Note**: This template installs dependencies directly rather than using `pip install -e ".[dev]"` because it's structured as a template repository, not an installable Python package.

3. (Optional) Install pre-commit hooks:
```bash
make pre-commit-install
```

### Try it! It's simple
```bash
python main.py
```

### Configuration

Configuration files are located in the `config/` directory:
- `defaults.yaml`: Main defaults file that imports all configs
- `model.yaml`: Model architecture parameters
- `training.yaml`: Training hyperparameters
- `wandb.yaml`: Wandb project settings
- `config.yaml`: General experiment settings

Override config values via command line:
```bash
python main.py training.learning_rate=0.01 training.batch_size=64
```

### Development

- **Lint code**: `make lint`
- **Format code**: `make format`
- **Run tests**: `make test`
- **Clean cache**: `make clean`

## Project Structure

```
roboml-research-template/
├── config/              # Hydra configuration files
│   ├── config_schema.py # Dataclass schemas
│   ├── defaults.yaml    # Main defaults
│   ├── model.yaml       # Model config
│   ├── training.yaml    # Training config
│   ├── wandb.yaml       # Wandb config
│   └── config.yaml      # General config
├── networks/            # Network architectures
├── learner/             # Learning algorithms
├── tests/               # Test files
└── main.py              # Entry point
```



## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

See LICENSE file for details.
