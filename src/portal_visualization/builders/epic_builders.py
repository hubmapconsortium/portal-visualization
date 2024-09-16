from abc import abstractmethod
from vitessce import VitessceConfig, ObsSegmentationsOmeTiffWrapper, AnnDataWrapper, \
    get_initial_coordination_scope_prefix, CoordinationLevel as CL
from .base_builders import ConfCells
from ..utils import get_conf_cells
from .base_builders import ViewConfBuilder
from requests import get
import re

from ..paths import OFFSETS_DIR, IMAGE_PYRAMID_DIR

zarr_path = 'hubmap_ui/seg-to-mudata-zarr/secondary_analysis.zarr'

# EPIC builders take in a vitessce conf output by a previous builder and modify it
# accordingly to add the EPIC-specific configuration.


class EPICConfBuilder(ViewConfBuilder):
    def __init__(self, epic_uuid, base_conf: ConfCells, entity, groups_token, assets_endpoint, **kwargs) -> None:
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)

        conf, cells = base_conf

        if conf is None:
            raise ValueError("ConfCells object must have a conf attribute")

        self._is_plural = isinstance(conf, list)

        if self._is_plural:
            self._base_conf = [
                VitessceConfig.from_dict(conf) for conf in conf
            ]
        else:
            self._base_conf: VitessceConfig = VitessceConfig.from_dict(base_conf.conf)

        self._epic_uuid = epic_uuid
        pass

    def get_conf_cells(self):
        self.apply()
        if (self._is_plural):
            return get_conf_cells([conf.to_dict() for conf in self._base_conf])
        return get_conf_cells(self._base_conf)

    def apply(self):
        if self._is_plural:
            for conf in self._base_conf:
                self._apply(conf)
        else:
            self._apply(self._base_conf)

    @abstractmethod
    def _apply(self, conf):  # pragma: no cover
        pass

    def zarr_store_url(self):
        adata_url = self._build_assets_url(zarr_path, use_token=False)
        return adata_url

    def segmentations_url(self, img_path):
        img_url = self._build_assets_url(img_path)
        return (
            img_url,
            str(
                re.sub(
                    r"ome\.tiff?",
                    "offsets.json",
                    re.sub(IMAGE_PYRAMID_DIR, OFFSETS_DIR, img_url),
                )
            ),
        )


class SegmentationMaskBuilder(EPICConfBuilder):
    def _apply(self, conf):
        zarr_url = self.zarr_store_url()
        datasets = conf.get_datasets()
        # TODO: add the correct path to the segmentation mask ome-tiff (image-pyramid)
        seg_path = f'{self.segmentations_url("seg")}/'
        # print(seg_path)
        seg_path = (
            'https://assets.hubmapconsortium.org/c9d9ab5c9ee9642b60dd351024968627/'
            'ometiff-pyramids/VAN0042-RK-3-18-registered-PAS-to-postAF-registered.ome_mask.ome.tif?'
            'token=AgzQXm7nvOW32vWw0EPpKonwbOqjNBzNvvW1p15855NoYglJxyfkC8rlJJWy8V6E8MeyXOwlpKdNBnHb5qnv7f8oeeG'
        )
        mask_names = self.read_metadata_from_url()
        mask_names = ['mask1', 'mask2']  # for testing purposes
        if (mask_names is not None):
            segmentation_objects = create_segmentation_objects(zarr_url, mask_names)
            segmentations = ObsSegmentationsOmeTiffWrapper(
                img_url=seg_path,
                obs_types_from_channel_names=True,
                coordination_values={
                    "fileUid": "segmentation-mask"
                }
            )

            for dataset in datasets:
                dataset.add_object(segmentations)
                for obj in segmentation_objects:
                    dataset.add_object(obj)

                # TODO: what happens if these views already exist , and if there are other views, how to place these?
                spatial_view = conf.add_view("spatialBeta", dataset=dataset, w=8, h=12)
                lc_view = conf.add_view("layerControllerBeta", dataset=dataset, w=4, h=12, x=8, y=0)
                # without add_view can't access the metaCoordincatinSpace
                # (e.g. get_coordination_scope() https://python-docs.vitessce.io/api_config.html?
                # highlight=coordination#vitessce.config.VitessceChainableConfig.get_coordination_scope)
                conf.link_views_by_dict([spatial_view, lc_view], {
                    "segmentationLayer": CL([
                        {
                            "fileUid": "segmentation-mask",
                            "spatialLayerVisible": True,
                            "spatialLayerOpacity": 1,
                        }
                    ])

                }, meta=True, scope_prefix=get_initial_coordination_scope_prefix("A", "obsSegmentations"))

    def read_metadata_from_url(self):
        url = f'{self.zarr_store_url()}/metadata.json'
        print(f"metadata.json URL: {url}")
        # url ='https://portal.hubmapconsortium.org/browse/dataset/004d4f157df4ba07356cd805131dfc04.json'
        request_init = self._get_request_init() or {}
        response = get(url, **request_init)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "mask_name" in data:
                mask_name = data["mask_name"]
                print(f"Mask name found: {mask_name}")
                return mask_name
            else:
                print("'mask_name' key not found in the response.")
                return None
        else:
            # raise Exception(f"Failed to retrieve data: {response.status_code} - {response.reason}")
            pass  # for testing purposes


def create_segmentation_objects(base_url, mask_names):
    segmentation_objects = []
    for mask_name in mask_names:
        mask_url = f'{base_url}/{mask_name}.zarr'
        segmentations_zarr = AnnDataWrapper(
            adata_url=mask_url,
            obs_locations_path="obsm/X_spatial",
            obs_labels_names=mask_name,
            coordination_values={
                "obsType": mask_name
            }
        )
        segmentation_objects.append(segmentations_zarr)

    return segmentation_objects
