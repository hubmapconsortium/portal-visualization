# AI Coding Agent Instructions for portal-visualization

## Project Overview

This is a **HuBMAP visualization configuration generator** that converts dataset metadata into Vitessce viewer configurations. The library is used by both `portal-ui` and `search-api` to dynamically generate interactive visualizations for biological datasets (imaging, RNA-seq, ATAC-seq, etc.).

**Core Workflow**: Dataset entity JSON → `builder_factory.py` selects appropriate Builder → Builder generates Vitessce config → Rendered in portal or returned via API

## Architecture & Key Concepts

### Builder Factory Pattern

- **Entry point**: `builder_factory.get_view_config_builder(entity, get_entity, parent, epic_uuid)`
- **Selection logic**: Uses `vitessce-hints` metadata array from entity to determine which builder class to instantiate
  - Hints like `["is_image", "rna", "spatial"]` → `SpatialMultiomicAnnDataZarrViewConfBuilder`
  - Hints like `["is_image", "codex"]` → `StitchedCytokitSPRMViewConfBuilder`
  - See `builder_factory.py:process_hints()` for the full decision tree
- **Visualization lifting**: Image pyramids are "vis-lifted" from support datasets to their parent dataset pages via `parent` parameter

### Builder Hierarchy

All builders inherit from `ViewConfBuilder` (abstract base in `builders/base_builders.py`):

- **Core method**: `get_conf_cells(**kwargs)` returns `ConfCells(conf_dict, notebook_cells)` namedtuple
- **Common utilities**:
  - `_build_assets_url(rel_path)` constructs authenticated asset URLs with token params
  - `_get_request_init()` provides auth headers for Zarr stores (non-public data requires Bearer token)
- **Imaging builders**: Extend `AbstractImagingViewConfBuilder` for OME-TIFF pyramid handling
- **AnnData builders**: Handle Zarr-backed AnnData stores for sequencing data

### File Path Conventions

Builders discover data files using regex patterns defined in `paths.py`:

- Image pyramids: `stitched/expressions/` or `stitched_expressions/`
- Segmentation masks: `segmentation_masks_Probabilities_*` or `kaggle_mask/`
- Offsets: `output_offsets/*.offsets.json` (for optimized image loading via Viv)
- Image metadata: `image_metadata/*.metadata.json` (physical size units for scaling)

## Development Workflows

### Testing

Run tests via `./test.sh` which:

1. Validates README matches `vis-preview.py --help` output (docs must stay in sync!)
2. Runs `ruff` linting
3. Executes pytest with **100% coverage requirement** (`--doctest-modules` enabled)

Fixture structure:

- `test/good-fixtures/BuilderName/uuid-entity.json` → fixtures for valid datasets
- `test/bad-fixtures/uuid-entity.json` → error case testing
- `test/assaytype-fixtures/uuid.json` → mock assay type metadata

### Adding New Assay Support

1. Define assay constant in `assays.py` (e.g., `SEQFISH = "seqFish"`)
2. Create builder class in appropriate `builders/*_builders.py` file
3. Update `builder_factory.py:get_view_config_builder()` decision tree with new hint combinations
4. Add test fixtures: `test/good-fixtures/YourBuilder/{uuid}-entity.json`
5. Verify README describes when the builder is used (see "Imaging Data" section)

## Code Conventions

### Doctests

Inline doctests are mandatory for coverage. Use this pattern:

```python
def _build_assets_url(self, rel_path):
    """Create a url for an asset.
    >>> builder = _DocTestBuilder(
    ...   entity={"uuid": "uuid"}, groups_token='token',
    ...   assets_endpoint='https://example.com')
    >>> builder._build_assets_url("path/to/file.tiff")
    'https://example.com/uuid/path/to/file.tiff?token=token'
    """
```

### Error Handling

- Use `# pragma: no cover` for production-only code (e.g., Flask abort calls in `client.py`)
- Wrap builder errors in `ConfCells` with error message for graceful degradation in portal UI
- Log errors via `current_app.logger.error()` when Flask context available

## Critical Integration Points

### Portal-UI Integration

Called from `portal-ui/context/app/routes_browse.py`:

```python
from portal_visualization.builder_factory import get_view_config_builder
builder = get_view_config_builder(entity, get_entity_fn)
conf_cells = builder.get_conf_cells(marker=marker)
```

### Search-API Integration

Similar usage but may specify `minimal=True` kwarg for lightweight configs

### Environment-Specific URLs

`defaults.json` defines dev/prod endpoints:

- Assets: `https://assets.{dev.}hubmapconsortium.org`
- Entity API: `https://entity-api.{dev.}hubmapconsortium.org`
- Always use `assets_endpoint` parameter, never hardcode URLs

## Dependencies & Versioning

- **Primary dependency**: `vitessce==3.7.4` (pinned due to downstream conflicts)
- **Release process**: Bump `VERSION.txt` → git tag → GitHub release → update `requirements.txt` in portal-ui and search-api
- **Python version**: Requires >=3.10 (see `setup.cfg`)

## Common Pitfalls

1. **Forgetting token auth**: Non-public datasets require `groups_token` in URLs or request headers
2. **Image pyramid detection**: Use `get_found_images()` from `utils.py`, not custom regex (handles `separate/` exclusions)
3. **Physical size scaling**: When overlaying segmentation masks, retrieve metadata JSONs and compute scale via `get_image_scale()` in `utils.py`
4. **Hint processing**: Add new hints to BOTH `process_hints()` return tuple AND the decision tree in `get_view_config_builder()`
