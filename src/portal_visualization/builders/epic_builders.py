from abc import abstractmethod
from vitessce import VitessceConfig, ObsSegmentationsOmeTiffWrapper, AnnDataWrapper, \
    get_initial_coordination_scope_prefix, CoordinationLevel as CL
from .base_builders import ConfCells
from ..utils import get_conf_cells, get_matches
from .base_builders import ViewConfBuilder
from requests import get
import re
import random
from ..paths import OFFSETS_DIR, IMAGE_PYRAMID_DIR, SEGMENTATION_SUBDIR, SEGMENTATION_ZARR_STORES, \
    SEGMENTATION_DIR, SEGMENTATION_SUPPORT_IMAGE_SUBDIR

transformations_filename = 'transformations.json'
zarr_path = f'{SEGMENTATION_SUBDIR}/{SEGMENTATION_ZARR_STORES}'

# EPIC builders take in a vitessce conf output by a previous builder and modify it
# accordingly to add the EPIC-specific configuration.


class EPICConfBuilder(ViewConfBuilder):
    def __init__(self, epic_uuid, base_conf: ConfCells, entity,
                 groups_token, assets_endpoint, **kwargs) -> None:
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)

        conf, cells = base_conf

        if conf is None:  # pragma: no cover
            raise ValueError("ConfCells object must have a conf attribute")
        # Not sure if need this, as assumption is to have 1 base image
        self._is_plural = isinstance(conf, list)

        if self._is_plural:  # pragma: no cover
            self._base_conf = [
                VitessceConfig.from_dict(conf) for conf in conf
            ]
        else:
            self._base_conf: VitessceConfig = VitessceConfig.from_dict(base_conf.conf)

        self._epic_uuid = epic_uuid

        pass

    def get_conf_cells(self):
        self.apply()
        if (self._is_plural):  # pragma: no cover
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

    def image_transofrmations_url(self):  # pragma: no cover
        transformations_url = self._build_assets_url(SEGMENTATION_DIR, use_token=True)
        return transformations_url

    def segmentations_ome_offset_url(self, img_path):
        img_url = self._build_assets_url(f'{SEGMENTATION_SUBDIR}/{img_path}')
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
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]
        found_images = [
            path for path in get_matches(
                file_paths_found, IMAGE_PYRAMID_DIR + r".*\.ome\.tiff?$",
            )
        ]
        # Remove the base-image pyramids from the found_images
        filtered_images = [img_path for img_path in found_images if SEGMENTATION_SUPPORT_IMAGE_SUBDIR not in img_path]
        if len(filtered_images) == 0:  # pragma: no cover
            message = f"Image pyramid assay with uuid {self._uuid} has no matching files"
            raise FileNotFoundError(message)

        elif len(filtered_images) >= 1:
            img_url, offsets_url = self.segmentations_ome_offset_url(
                filtered_images[0]
            )

        segmentation_scale = self.read_segmentation_scale()
        segmentations = ObsSegmentationsOmeTiffWrapper(
            img_url=img_url,
            offsets_url=offsets_url,
            coordinate_transformations=[{"type": "scale", "scale": segmentation_scale}],
            obs_types_from_channel_names=True,
            coordination_values={
                "fileUid": "segmentation-mask"
            }
        )

        mask_names = self.read_metadata_from_url()
        if (mask_names is not None):  # pragma: no cover
            segmentation_objects, segmentations_CL = create_segmentation_objects(zarr_url, mask_names)
            for dataset in datasets:
                dataset.add_object(segmentations)
                for obj in segmentation_objects:
                    dataset.add_object(obj)

            spatial_view = conf.get_first_view_by_type('spatialBeta')
            lc_view = conf.get_first_view_by_type('layerControllerBeta')
            conf.link_views_by_dict([spatial_view, lc_view], {
                # Neutralizing the base-image colors
                'imageLayer': CL([{'photometricInterpretation': 'RGB', }]),
                "segmentationLayer": CL([
                    {
                        "fileUid": "segmentation-mask",
                        "spatialLayerVisible": True,
                        "spatialLayerOpacity": 1,
                        "segmentationChannel": CL(segmentations_CL)
                    }
                ])

            }, meta=True, scope_prefix=get_initial_coordination_scope_prefix("A", "obsSegmentations"))

    def read_metadata_from_url(self):  # pragma: no cover
        mask_names = []
        url = f'{self.zarr_store_url()}/metadata.json'
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

    def read_segmentation_scale(self):  # pragma: no cover
        url = self._build_assets_url(f'{SEGMENTATION_DIR}/{transformations_filename}')
        request_init = self._get_request_init() or {}
        # By default no scaling should be applied, format accepted by vitessce
        scale = [1, 1, 1, 1, 1]
        response = get(url, **request_init)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "scale" in data:
                scale = data["scale"]
            else:
                print("'scale' key not found in the response.")
        else:
            print(f"Failed to retrieve {transformations_filename}: {response.status_code} - {response.reason}")
        return scale


def create_segmentation_objects(base_url, mask_names):  # pragma: no cover
    spatialTargetCMapping = {
        'arteries-arterioles': 4,
        'glomeruli': 2,
        'tubules': 3,
    }
    segmentation_objects = []
    segmentations_CL = []
    for index, mask_name in enumerate(mask_names):
        color_channel = generate_unique_color()
        mask_url = f'{base_url}/{mask_name}.zarr'
        segmentations_zarr = AnnDataWrapper(
            adata_url=mask_url,
            obs_locations_path="obsm/X_spatial",
            obs_labels_names=mask_name,
            coordination_values={
                "obsType": mask_name
            }
        )
        # TODO: manually adjusted for the test dataset, need to be fixed on Vitessce side
        if all(mask in mask_names for mask in spatialTargetCMapping.keys()):
            channelIndex = spatialTargetCMapping[mask_name]
        else:
            channelIndex = index
        seg_CL = {
            "spatialTargetC": channelIndex,
            "obsType": mask_name,
            "spatialChannelOpacity": 1,
            "spatialChannelColor": color_channel,
            "obsHighlight": None

        }
        segmentation_objects.append(segmentations_zarr)
        segmentations_CL.append(seg_CL)
    return segmentation_objects, segmentations_CL


def generate_unique_color():  # pragma: no cover
    return [random.randint(0, 255) for _ in range(3)]
