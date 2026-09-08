# Units used in the image metadata for physical sizes
image_units = {"nm": 1e9, "μm": 1e6, "mm": 1e3, "cm": 1e2, "dm": 10}

# The base image pyramids for kaggle-1 and kaggle-2 may have various directory structures depending
#  upon when they were processed. For older datasets, the image pyramids will be present
#  either in 'processed_microscopy', or 'processedMicroscopy' while newer datasets will be listed under lab_processed.
base_image_dirs = ["lab_processed", "processed_microscopy", "processedMicroscopy"]

ZARR_PATH = "hubmap_ui/anndata-zarr/secondary_analysis.zarr"
ZIP_ZARR_PATH = f"{ZARR_PATH}.zip"
MULTIOMIC_ZARR_PATH = "hubmap_ui/mudata-zarr/secondary_analysis.zarr"
XENIUM_ZARR_PATH = "Xenium.zarr"

# Maximum number of observations to display heatmaps for performance reasons
MAX_OBS_FOR_HEATMAP = 125_000

# User-Agent sent on server-side requests made while building a Vitessce config. It must NOT match
# the back-end's scraping filter, so the config builder's traffic can be whitelisted:
# ~*(?i)(aiohttp|python-httpx|python-requests|Python-urllib)
PORTAL_VIS_USER_AGENT = "portal-visualization-config-builder"
