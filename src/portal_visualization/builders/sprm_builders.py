import re
from pathlib import Path

import zarr
from vitessce import (
    AnnDataWrapper,
    CoordinationType,
    MultiImageWrapper,
    OmeTiffWrapper,
    VitessceConfig,
)
from vitessce import (
    Component as cm,
)
from vitessce import (
    FileType as ft,
)

from ..paths import (
    CODEX_TILE_DIR,
    IMAGE_PYRAMID_DIR,
    SPRM_JSON_DIR,
    SPRM_PYRAMID_SUBDIR,
    STITCHED_IMAGE_DIR,
    STITCHED_REGEX,
    TILE_REGEX,
)
from ..utils import create_coordination_values, get_conf_cells, get_matches, read_zip_zarr
from .base_builders import ViewConfBuilder
from .imaging_builders import ImagePyramidViewConfBuilder

# https://github.com/hubmapconsortium/portal-containers/blob/master/containers/sprm-to-anndata
# has information on how these keys are generated.
DEFAULT_SPRM_ANNDATA_FACTORS = [
    "Cell K-Means [tSNE_All_Features]",
    "Cell K-Means [Mean-All-SubRegions] Expression",
    "Cell K-Means [Mean] Expression",
    "Cell K-Means [Shape-Vectors]",
    "Cell K-Means [Texture]",
    "Cell K-Means [Total] Expression",
    "Cell K-Means [Covariance] Expression",
]


class CytokitSPRMViewConfigError(Exception):
    """Raised when one of the individual SPRM view configs errors out for Cytokit"""


class SPRMViewConfBuilder(ImagePyramidViewConfBuilder):
    """Base class with shared methods for different SPRM subclasses,
    like SPRMJSONViewConfBuilder and SPRMAnnDataViewConfBuilder
    https://portal.hubmapconsortium.org/search?mapped_data_types[0]=CODEX%20%5BCytokit%20%2B%20SPRM%5D&entity_type[0]=Dataset
    """

    def _get_full_image_path(self):
        return f"{self._imaging_path_regex}/{self._image_name}" + r"\.ome\.tiff?"

    def _check_sprm_image(self, path_regex):
        """Check whether or not there is a matching SPRM image at a path.
        :param str path_regex: The path to look for the images
        :rtype: str The found image
        """
        file_paths_found = self._get_file_paths()
        found_image_files = get_matches(file_paths_found, path_regex)
        if len(found_image_files) != 1:  # pragma: no cover
            message = f'Found {len(found_image_files)} image files for SPRM uuid "{self._uuid}".'
            raise FileNotFoundError(message)
        found_image_file = found_image_files[0]
        return found_image_file

    def _get_ometiff_image_wrapper(self, found_image_file, found_image_path):
        """Create a OmeTiffWrapper object for an image, including offsets.json after calling
        _get_img_and_offset_url on the arguments to this function.
        :param str found_image_file: The path to look for the image itself
        :param str found_image_path: The folder to be replaced with the offsets path
        """
        img_url, offsets_url, _ = self._get_img_and_offset_url(
            found_image_file,
            re.escape(found_image_path),
        )
        return OmeTiffWrapper(img_url=img_url, offsets_url=offsets_url, name=self._image_name)


class SPRMJSONViewConfBuilder(SPRMViewConfBuilder):
    """Wrapper class for generating "first generation" non-stitched JSON-backed
    SPRM Vitessce configurations, like
    https://portal.hubmapconsortium.org/browse/dataset/dc31a6d06daa964299224e9c8d6cafb3
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        # All "file" Vitessce objects that do not have wrappers.
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        # These are both something like R001_X009_Y009 because
        # there is no mask used here or shared name with the mask data.
        self._base_name = kwargs["base_name"]
        self._image_name = kwargs["base_name"]
        self._imaging_path_regex = kwargs["imaging_path"]
        self._files = [
            {
                "rel_path": f"{SPRM_JSON_DIR}/" + f"{self._base_name}.cells.json",
                "file_type": ft.CELLS_JSON,
                "coordination_values": create_coordination_values(),
            },
            {
                "rel_path": f"{SPRM_JSON_DIR}/" + f"{self._base_name}.cell-sets.json",
                "file_type": ft.CELL_SETS_JSON,
                "coordination_values": create_coordination_values(),
            },
            {
                "rel_path": f"{SPRM_JSON_DIR}/" + f"{self._base_name}.clusters.json",
                "file_type": "clusters.json",
                "coordination_values": create_coordination_values(),
            },
        ]

    def get_conf_cells(self, **kwargs):
        found_image_file = self._check_sprm_image(self._get_full_image_path())
        vc = VitessceConfig(name=self._base_name, schema_version=self._schema_version)
        dataset = vc.add_dataset(name="SPRM")
        image_wrapper = self._get_ometiff_image_wrapper(found_image_file, self._imaging_path_regex)
        dataset = dataset.add_object(image_wrapper)
        file_paths_found = self._get_file_paths()
        if self._files[0]["rel_path"] not in file_paths_found:
            # This tile has no segmentations,
            # so only show Spatial component without cells sets, genes etc.
            vc = self._setup_view_config(vc, dataset, self.view_type, disable_3d=[self._image_name])
        else:
            # This tile has segmentations so show the analysis results.
            for file in self._files:
                path = file["rel_path"]
                if path not in file_paths_found:
                    message = f'SPRM file {path} with uuid "{self._uuid}" not found as expected.'
                    raise FileNotFoundError(message)
                dataset_file = self._replace_url_in_file(file)
                dataset = dataset.add_file(**(dataset_file))
            vc = self._setup_view_config_raster_cellsets_expression_segmentation(vc, dataset)
        return get_conf_cells(vc)

    def _setup_view_config_raster_cellsets_expression_segmentation(self, vc, dataset):
        vc.add_view(cm.SPATIAL, dataset=dataset, x=3, y=0, w=7, h=8)
        vc.add_view(cm.DESCRIPTION, dataset=dataset, x=0, y=8, w=3, h=4)
        vc.add_view(cm.LAYER_CONTROLLER, dataset=dataset, x=0, y=0, w=3, h=8).set_props(disable3d=[self._image_name])
        vc.add_view(cm.OBS_SETS, dataset=dataset, x=10, y=5, w=2, h=7)
        vc.add_view(cm.FEATURE_LIST, dataset=dataset, x=10, y=0, w=2, h=5).set_props(variablesLabelOverride="antigen")
        vc.add_view(cm.HEATMAP, dataset=dataset, x=3, y=8, w=7, h=4).set_props(
            transpose=True, variablesLabelOverride="antigen"
        )
        return vc


class SPRMAnnDataViewConfBuilder(SPRMViewConfBuilder):
    """Wrapper class for generating "second generation"
    stitched AnnData-backed SPRM Vitessce configurations,
    like the dataset derived from
    https://portal.hubmapconsortium.org/browse/dataset/1c33472c68c4fb40f531b39bf6310f2d

    :param \\*\\*kwargs: { imaging_path: str, mask_path: str } for the paths
    of the image and mask relative to image_pyramid_regex
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._base_name = kwargs["base_name"]
        self._mask_name = kwargs["mask_name"]
        self._image_name = kwargs["image_name"]
        self._imaging_path_regex = f"{self.image_pyramid_regex}/{kwargs['imaging_path']}"
        self._mask_path_regex = f"{self.image_pyramid_regex}/{kwargs['mask_path']}"
        self._is_zarr_zip = False

    def zarr_store(self):
        zarr_path = f"anndata-zarr/{self._image_name}-anndata.zarr"
        zip_zarr_path = f"{zarr_path}.zip"
        request_init = self._get_request_init() or {}
        if self._is_zarr_zip:  # pragma: no cover
            adata_url = self._build_assets_url(zip_zarr_path, use_token=True)
            try:
                return read_zip_zarr(adata_url, request_init)
            except Exception as e:
                print(f"Error opening the zip zarr file. {e}")
        else:
            adata_url = self._build_assets_url(zarr_path, use_token=False)
            return zarr.open(adata_url, mode="r", storage_options={"client_kwargs": request_init})

    def _get_bitmask_image_path(self):
        return f"{self._mask_path_regex}/{self._mask_name}" + r"\.ome\.tiff?"

    def _get_ometiff_mask_wrapper(self, found_bitmask_file):
        bitmask_img_url, bitmask_offsets_url, _ = self._get_img_and_offset_url(
            found_bitmask_file,
            self.image_pyramid_regex,
        )
        return OmeTiffWrapper(
            img_url=bitmask_img_url, offsets_url=bitmask_offsets_url, name=self._mask_name, is_bitmask=True
        )

    def get_conf_cells(self, marker=None):
        vc = VitessceConfig(name=self._image_name, schema_version=self._schema_version)
        dataset = vc.add_dataset(name="SPRM")
        file_paths_found = self._get_file_paths()
        zarr_path = f"anndata-zarr/{self._image_name}-anndata.zarr"
        # Use the group as a proxy for presence of the rest of the zarr store.
        if f"{zarr_path}.zip" in file_paths_found:  # pragma: no cover
            self._is_zarr_zip = True
            zarr_path = f"{zarr_path}.zip"
        elif f"{zarr_path}/.zgroup" not in file_paths_found:  # pragma: no cover
            message = f"SPRM assay with uuid {self._uuid} has no .zarr store at {zarr_path}"
            raise FileNotFoundError(message)
        adata_url = self._build_assets_url(zarr_path, use_token=False)

        additional_cluster_names = list(self.zarr_store().get("uns/cluster_columns", []))

        obs_set_names = sorted(set(additional_cluster_names + DEFAULT_SPRM_ANNDATA_FACTORS))
        obs_set_paths = [f"obs/{key}" for key in obs_set_names]

        anndata_wrapper = AnnDataWrapper(
            adata_url=adata_url,
            is_zip=self._is_zarr_zip,
            obs_feature_matrix_path="X",
            obs_embedding_paths=["obsm/tsne"],
            obs_embedding_names=["t-SNE"],
            obs_set_names=obs_set_names,
            obs_set_paths=obs_set_paths,
            obs_locations_path="obsm/xy",
            request_init=self._get_request_init(),
        )
        dataset = dataset.add_object(anndata_wrapper)
        found_image_file = self._check_sprm_image(self._get_full_image_path())
        image_wrapper = self._get_ometiff_image_wrapper(found_image_file, self.image_pyramid_regex)
        found_bitmask_file = self._check_sprm_image(self._get_bitmask_image_path())
        bitmask_wrapper = self._get_ometiff_mask_wrapper(found_bitmask_file)
        dataset = dataset.add_object(MultiImageWrapper([image_wrapper, bitmask_wrapper]))
        vc = self._setup_view_config_raster_cellsets_expression_segmentation(vc, dataset, marker)
        return get_conf_cells(vc)

    def _setup_view_config_raster_cellsets_expression_segmentation(self, vc, dataset, marker):
        description = vc.add_view(cm.DESCRIPTION, dataset=dataset, x=0, y=8, w=3, h=4)
        layer_controller = vc.add_view(cm.LAYER_CONTROLLER, dataset=dataset, x=0, y=0, w=3, h=8)

        spatial = vc.add_view(cm.SPATIAL, dataset=dataset, x=3, y=0, w=4, h=8)
        scatterplot = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping="t-SNE", x=7, y=0, w=3, h=8)
        cell_sets = vc.add_view(cm.OBS_SETS, dataset=dataset, x=10, y=5, w=2, h=7)

        gene_list = vc.add_view(cm.FEATURE_LIST, dataset=dataset, x=10, y=0, w=2, h=5).set_props(
            variablesLabelOverride="antigen"
        )
        heatmap = vc.add_view(cm.HEATMAP, dataset=dataset, x=3, y=8, w=7, h=4).set_props(
            variablesLabelOverride="antigen", transpose=True
        )

        views = [description, layer_controller, spatial, cell_sets, gene_list, scatterplot, heatmap]

        if marker:
            vc.link_views(
                views,
                [CoordinationType.FEATURE_SELECTION, CoordinationType.OBS_COLOR_ENCODING],
                [[marker], "geneSelection"],
            )

        return vc


class MultiImageSPRMAnndataViewConfigError(Exception):
    """Raised when one of the individual SPRM view configs errors out"""


class MultiImageSPRMAnndataViewConfBuilder(ViewConfBuilder):
    """Wrapper class for generating multiple "second generation" AnnData-backed SPRM
    Vitessce configurations via SPRMAnnDataViewConfBuilder,
    used for datasets with multiple regions.
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._expression_id = "expr"
        self._mask_id = "mask"
        self._image_pyramid_subdir = SPRM_PYRAMID_SUBDIR
        self._mask_pyramid_subdir = SPRM_PYRAMID_SUBDIR.replace(self._expression_id, self._mask_id)

    def _find_ids(self):
        """Search the image pyramid directory for all of the names of OME-TIFF files
        to use as unique identifiers.
        """
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]
        full_pyramid_path = IMAGE_PYRAMID_DIR + "/" + self._image_pyramid_subdir
        pyramid_files = [file for file in file_paths_found if full_pyramid_path in file]
        found_ids = [
            Path(image_path)
            .name.replace(".ome.tiff", "")
            .replace(".ome.tif", "")
            .replace("_" + self._expression_id, "")
            for image_path in pyramid_files
        ]
        if len(found_ids) == 0:
            raise FileNotFoundError(f"Could not find images of the SPRM analysis with uuid {self._uuid}")
        return found_ids

    def get_conf_cells(self, marker=None):
        found_ids = self._find_ids()
        confs = []
        for id in sorted(found_ids):
            builder = SPRMAnnDataViewConfBuilder(
                entity=self._entity,
                groups_token=self._groups_token,
                assets_endpoint=self._assets_endpoint,
                base_name=id,
                imaging_path=self._image_pyramid_subdir,
                mask_path=self._mask_pyramid_subdir,
                image_name=f"{id}_{self._expression_id}",
                mask_name=f"{id}_{self._mask_id}",
            )
            conf = builder.get_conf_cells(marker=marker).conf
            if conf == {}:
                raise MultiImageSPRMAnndataViewConfigError(  # pragma: no cover
                    f"Cytokit SPRM assay with uuid {self._uuid} has empty view\
                        config for id '{id}'"
                )
            confs.append(conf)
        conf = confs if len(confs) > 1 else confs[0]
        return get_conf_cells(conf)


class StitchedCytokitSPRMViewConfBuilder(MultiImageSPRMAnndataViewConfBuilder):
    """Wrapper class for generating multiple "second generation" stitched AnnData-backed SPRM
    Vitessce configurations via SPRMAnnDataViewConfBuilder,
    used for datasets with multiple regions.
    These are from post-August 2020 Cytokit datasets (stitched).
    """

    # Need to override base class settings due to different directory structure
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._image_pyramid_subdir = STITCHED_IMAGE_DIR
        # The ids don't match exactly with the replacement because all image files have
        # stitched_expressions appended while the subdirectory only has /stitched/
        self._expression_id = "stitched_expressions"
        self._mask_pyramid_subdir = STITCHED_IMAGE_DIR.replace("expressions", "mask")
        self._mask_id = "stitched_mask"


class TiledSPRMViewConfBuilder(ViewConfBuilder):
    """Wrapper class for generating many "first generation"
    non-stitched JSON-backed SPRM Vitessce configurations,
    one per tile per region, via SPRMJSONViewConfBuilder.
    """

    def get_conf_cells(self, **kwargs):
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]
        found_tiles = get_matches(file_paths_found, TILE_REGEX) or get_matches(file_paths_found, STITCHED_REGEX)
        if len(found_tiles) == 0:  # pragma: no cover
            message = f"Cytokit SPRM assay with uuid {self._uuid} has no matching tiles"
            raise FileNotFoundError(message)
        confs = []
        for tile in sorted(found_tiles):
            builder = SPRMJSONViewConfBuilder(
                entity=self._entity,
                groups_token=self._groups_token,
                assets_endpoint=self._assets_endpoint,
                base_name=tile,
                imaging_path=CODEX_TILE_DIR,
            )
            conf = builder.get_conf_cells().conf
            if conf == {}:  # pragma: no cover
                message = f"Cytokit SPRM assay with uuid {self._uuid} has empty view config"
                raise CytokitSPRMViewConfigError(message)
            confs.append(conf)
        return get_conf_cells(confs)
