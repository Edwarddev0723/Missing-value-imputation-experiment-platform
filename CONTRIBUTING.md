# Contributing to ImputeBench

Thank you for your interest in contributing! Here's how you can help.

## Ways to Contribute

- **Add a new imputation method** — see [docs/adding-methods.md](docs/adding-methods.md)
- **Add a new dataset** — see [docs/adding-datasets.md](docs/adding-datasets.md)
- **Report bugs** — open an issue with reproduction steps
- **Improve documentation** — fix typos, add examples, clarify guides
- **Submit benchmark results** — run on new hardware/configurations and share results

## Development Setup

```bash
git clone https://github.com/yourusername/ImputeBench.git
cd ImputeBench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Code Style

- Python 3.10+ with type hints
- Formatted with [Ruff](https://github.com/astral-sh/ruff) (line length: 100)
- Run linter: `ruff check src/ tests/`

## Running Tests

```bash
pytest tests/ -v
```

All tests must pass before submitting a PR.

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Make your changes with clear commit messages.
3. Add or update tests if applicable.
4. Ensure all tests pass: `pytest tests/ -v`.
5. Run the linter: `ruff check src/ tests/`.
6. Submit a PR with a clear description of the changes.

## Adding a New Method — Checklist

- [ ] Implement class extending `BaseImputer` in `src/methods/{category}/`
- [ ] Register with `@METHOD_REGISTRY.register("name")`
- [ ] Add YAML config in `configs/methods/`
- [ ] Add import in `src/factory.py`
- [ ] Verify with `pytest tests/test_methods.py -v`
- [ ] Run at least one experiment end-to-end

## Adding a New Dataset — Checklist

- [ ] Implement class extending `BaseDataset` in `src/datasets/`
- [ ] Register with `@DATASET_REGISTRY.register("name")`
- [ ] Add YAML config in `configs/datasets/`
- [ ] Add import in `src/factory.py`
- [ ] Update `analysis/configs/dataset_track_mapping.yaml`
- [ ] Verify with `pytest tests/test_datasets.py -v`

## Code of Conduct

Be respectful and constructive. We're all here to advance research.
