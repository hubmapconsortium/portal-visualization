# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python library that converts HuBMAP dataset metadata into Vitessce visualization configurations. Used by `portal-ui` and `search-api`.

**Core flow**: Dataset entity JSON → `builder_factory.py` selects Builder → Builder generates Vitessce config

## Common Commands

```bash
# Run full test suite (lint + format check + pytest with 100% coverage)
source .venv/bin/activate && ./test.sh

# Run tests only
uv run pytest -vv --doctest-modules

# Run a single test file
uv run pytest test/test_builders.py -vv

# Run a single test
uv run pytest test/test_builders.py::TestClassName::test_name -vv

# Thin install tests only (no heavy deps)
uv run pytest -m "not requires_full" -vv --doctest-modules

# Lint and format
uv run ruff check src/ test/
uv run ruff check --fix src/ test/
uv run ruff format src/ test/

# Coverage
PYTHONPATH=. uv run coverage run --module pytest . -vv --doctest-modules
uv run coverage report --show-missing --fail-under 100

# Install for development
pip install -e ".[all]"   # or: uv sync --all-extras
```

## Architecture

### Dual Installation Modes

- **Thin install** (`pip install portal-visualization`): Zero dependencies, only provides `has_visualization()` and `process_hints()`. Used by services that just check if datasets have visualization support.
- **Full install** (`pip install portal-visualization[full]`): Includes vitessce, zarr, etc. Required for actual visualization generation.

This dual mode drives a key architectural constraint: **all builder imports must be lazy** (string-based references, imported only when needed) so the thin install never touches heavy dependencies.

### Builder Selection

`builder_registry.py` — declarative config with priority-based matching. Each builder has `required_hints`, `forbidden_hints`, `assay_types`, `parent_assay_types`, and `priority`. The registry returns a string builder name, resolved to a class via `_lazy_import_builder()` in `builder_factory.py`.

### Builder Hierarchy

All builders inherit from `ViewConfBuilder` in `builders/base_builders.py`. Core method is `get_conf_cells(**kwargs)` → `ConfCells(conf_dict, notebook_cells)` namedtuple. Imaging builders extend `AbstractImagingViewConfBuilder`; sequencing builders work with Zarr-backed AnnData stores.

### Source Layout

- `src/portal_visualization/` — main package (src-layout)
- `src/portal_visualization/builders/` — all builder implementations (~19 builder classes)
- `test/good-fixtures/BuilderName/` — per-builder test fixture directories
- `test/bad-fixtures/` — error case fixtures
- `test/assaytype-fixtures/` — mock assay type metadata

### Key Modules

- `builder_factory.py` — builder selection logic and `get_view_config_builder()` public API
- `builder_registry.py` — declarative registry-based builder selection
- `cli.py` — `vis-preview` CLI entry point
- `client.py` — `ApiClient` for portal-ui/search-api integration (Flask context)
- `paths.py` — regex patterns for discovering data files in datasets
- `utils.py` — image detection, zarr format detection, scale computation
- `data_access.py` — abstractions for resource loading and zarr access
- `view_layout.py` — layout configuration logic

## Key Constraints

- **100% test coverage required** (`fail_under = 100` in pyproject.toml). Use `# pragma: no cover` only for production-only code paths (e.g., Flask abort calls).
- **Doctests are mandatory** for coverage — `--doctest-modules` is enabled. Functions need inline doctests using `_DocTestBuilder` helper pattern.
- **README.md must match `vis-preview --help`** output — `test.sh` validates this automatically.
- **Ruff** for linting and formatting: line length 120, target py312, double quotes.
- Mark tests needing heavy dependencies with `@pytest.mark.requires_full`.
- Version is in `VERSION.txt` (single source of truth).
- Python `>=3.12` (`requires-python` in `pyproject.toml`).
- `vitessce==3.9.2` is pinned — in the `[full]` extra, not the core dependency list, which is empty by design.

## Adding a New Builder

1. Define assay constant in `assays.py` if needed
2. Create builder class in appropriate `builders/*_builders.py` file
3. Add config entry to `builder_registry.py:populate_registry()`
4. Add lazy import to `builder_factory.py:_lazy_import_builder()`
5. Add test fixtures in `test/good-fixtures/YourBuilder/{uuid}-entity.json`