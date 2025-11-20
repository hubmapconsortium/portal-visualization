"""Portal Visualization - HuBMAP Dataset Visualization Configuration Generator.

This package provides two install modes:

1. **Thin Install** (default): `pip install portal-visualization`
   - Provides only the has_visualization() and process_hints() functions
   - No heavy dependencies (vitessce, zarr, etc.)
   - Use for checking if a dataset has visualization support
   - Very lightweight install
   - Does not support visualization generation

2. **Full Install**: `pip install portal-visualization[full]`
   - Provides complete visualization generation capabilities
   - Includes all dependencies for Vitessce configuration generation
   - Use for actual visualization rendering
   - Heavier install due to additional dependencies

Example usage (thin install):
    >>> from portal_visualization import has_visualization
    >>> entity = {"uuid": "abc123", "vitessce-hints": []}
    >>> has_visualization(entity, lambda x: {})
    False
    >>> entity_2 = {"uuid": "def456", "vitessce-hints": ["rna", "atac"]}
    >>> has_visualization(entity_2, lambda x: {})
    True

Note: Check `client.py`'s get_vitessce_conf_cells_and_lifted_uuid for example usage of
the full install capabilities.
"""

# Expose lightweight functions that work with thin install
from .builder_factory import has_visualization, process_hints

__all__ = ["has_visualization", "process_hints"]

# pytest doctests fail without this.
