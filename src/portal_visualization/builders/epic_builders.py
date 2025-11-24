import random
import re
from abc import abstractmethod

from requests import get
from vitessce import (
    AnnDataWrapper,
    ObsSegmentationsOmeTiffWrapper,
    VitessceConfig,
    get_initial_coordination_scope_prefix,
)
from vitessce import CoordinationLevel as CL

from ..paths import (
    IMAGE_METADATA_DIR,
    IMAGE_PYRAMID_DIR,
    OFFSETS_DIR,
    SEGMENTATION_SUBDIR,
    SEGMENTATION_SUPPORT_IMAGE_SUBDIR,
    SEGMENTATION_ZARR_STORES,
)
from ..utils import get_conf_cells, get_image_metadata, get_image_scale, get_matches
from .base_builders import ConfCells, ViewConfBuilder

zarr_path = f"{SEGMENTATION_SUBDIR}/{SEGMENTATION_ZARR_STORES}"

# EPIC builders take in a vitessce conf output by a previous builder and modify it
# accordingly to add the EPIC-specific configuration.


class EPICConfBuilder(ViewConfBuilder):
    def __init__(
        self, epic_uuid, base_conf: ConfCells, entity, groups_token, assets_endpoint, base_image_metadata, **kwargs
    ) -> None:
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)

        conf, cells = base_conf

        if conf is None:  # pragma: no cover
            raise ValueError("ConfCells object must have a conf attribute")
        # Not sure if need this, as assumption is to have 1 base image
        self._is_plural = isinstance(conf, list)

        if self._is_plural:  # pragma: no cover
            self._base_conf = [VitessceConfig.from_dict(conf) for conf in conf]
        else:
            self._base_conf: VitessceConfig = VitessceConfig.from_dict(base_conf.conf)

        self._epic_uuid = epic_uuid
        self._is_zarr_zip = False
        self.base_image_metadata = base_image_metadata

    def get_conf_cells(self):
        self.apply()
        if self._is_plural:  # pragma: no cover
            return get_conf_cells([conf.to_dict() for conf in self._base_conf])
        return get_conf_cells(self._base_conf)

    def apply(self):
        if self._is_plural:  # pragma: no cover
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

    def segmentations_ome_offset_url(self, img_path):
        img_url = self._build_assets_url(f"{SEGMENTATION_SUBDIR}/{img_path}")
        return (
            img_url,
            str(
                re.sub(
                    r"ome\.tiff?",
                    "offsets.json",
                    re.sub(IMAGE_PYRAMID_DIR, OFFSETS_DIR, img_url),
                )
            ),
            str(
                re.sub(
                    r"ome\.tiff?",
                    "metadata.json",
                    re.sub(IMAGE_PYRAMID_DIR, IMAGE_METADATA_DIR, img_url),
                )
            ),
        )


class SegmentationMaskBuilder(EPICConfBuilder):
    def _apply(self, conf):
        zarr_url = self.zarr_store_url()
        datasets = conf.get_datasets()
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]
        if any(".zarr.zip" in path for path in file_paths_found):
            self._is_zarr_zip = True
        found_images = list(
            get_matches(
                file_paths_found,
                IMAGE_PYRAMID_DIR + r".*\.ome\.tiff?$",
            )
        )
        # Remove the base-image pyramids from the found_images
        filtered_images = [img_path for img_path in found_images if SEGMENTATION_SUPPORT_IMAGE_SUBDIR not in img_path]
        if len(filtered_images) == 0:  # pragma: no cover
            message = f"Image pyramid assay with uuid {self._uuid} has no matching files"
            raise FileNotFoundError(message)

        if len(filtered_images) >= 1:
            img_url, offsets_url, metadata_url = self.segmentations_ome_offset_url(filtered_images[0])
        segmentation_metadata = get_image_metadata(self, metadata_url)

        segmentation_scale = get_image_scale(self.base_image_metadata, segmentation_metadata)
        segmentations = ObsSegmentationsOmeTiffWrapper(
            img_url=img_url,
            offsets_url=offsets_url,
            coordinate_transformations=[{"type": "scale", "scale": segmentation_scale}],
            obs_types_from_channel_names=True,
            coordination_values={"fileUid": "segmentation-mask"},
        )

        mask_names = self.read_metadata_from_url()
        if mask_names is not None:  # pragma: no cover
            segmentation_objects, segmentations_CL = create_segmentation_objects(self, zarr_url, mask_names)
            for dataset in datasets:
                dataset.add_object(segmentations)
                for obj in segmentation_objects:
                    dataset.add_object(obj)

            spatial_view = conf.get_first_view_by_type("spatialBeta")
            lc_view = conf.get_first_view_by_type("layerControllerBeta")
            conf.link_views_by_dict(
                [spatial_view, lc_view],
                {
                    # Neutralizing the base-image colors
                    "imageLayer": CL(
                        [
                            {
                                "photometricInterpretation": "RGB",
                            }
                        ]
                    ),
                    "segmentationLayer": CL(
                        [
                            {
                                "fileUid": "segmentation-mask",
                                "spatialLayerVisible": True,
                                "spatialLayerOpacity": 1,
                                "segmentationChannel": CL(segmentations_CL),
                            }
                        ]
                    ),
                },
                meta=True,
                scope_prefix=get_initial_coordination_scope_prefix("A", "obsSegmentations"),
            )

    def read_metadata_from_url(self):  # pragma: no cover
        mask_names = []
        url = f"{self.zarr_store_url()}/metadata.json"
        request_init = self._get_request_init() or {}
        response = get(url, **request_init)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "mask_names" in data:
                mask_names = data["mask_names"]
            else:
                print("'mask_names' key not found in the response.")
        else:
            # in this case, the code won't execute for this
            print(f"Failed to retrieve metadata.json: {response.status_code} - {response.reason}")
        return mask_names


def create_segmentation_objects(self, base_url, mask_names):  # pragma: no cover
    segmentation_objects = []
    segmentations_CL = []
    for mask_name in mask_names:
        color_channel = generate_unique_color()
        mask_url = f"{base_url}/{mask_name}.zarr"
        if self._is_zarr_zip:
            mask_url = f"{mask_url}.zip"
        segmentations_zarr = AnnDataWrapper(
            adata_url=mask_url,
            is_zip=self._is_zarr_zip,
            obs_locations_path="obsm/X_spatial",
            obs_labels_names=mask_name,
            coordination_values={"obsType": mask_name},
        )
        seg_CL = {
            "spatialTargetC": mask_name,
            "obsType": mask_name,
            "spatialChannelOpacity": 1,
            "spatialChannelColor": color_channel,
            "obsHighlight": None,
        }
        segmentation_objects.append(segmentations_zarr)
        segmentations_CL.append(seg_CL)
    return segmentation_objects, segmentations_CL


def generate_unique_color():  # pragma: no cover
    return [random.randint(0, 255) for _ in range(3)]
