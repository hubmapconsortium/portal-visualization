import re
from pathlib import Path

from vitessce import (
    AnnDataWrapper,
    ImageOmeTiffWrapper,
    MultiImageWrapper,
    ObsSegmentationsOmeTiffWrapper,
    OmeTiffWrapper,
    VitessceConfig,
    get_initial_coordination_scope_prefix,
)
from vitessce import (
    Component as cm,
)
from vitessce import (
    CoordinationLevel as CL,
)

from ..constants import base_image_dirs
from ..paths import (
    GEOMX_DIR,
    IMAGE_METADATA_DIR,
    IMAGE_PYRAMID_DIR,
    OFFSETS_DIR,
    SEGMENTATION_SUBDIR,
    SEGMENTATION_SUPPORT_IMAGE_SUBDIR,
    SEQFISH_FILE_REGEX,
    SEQFISH_HYB_CYCLE_REGEX,
)
from ..utils import (
    get_conf_cells,
    get_found_images,
    get_found_images_all,
    get_image_metadata,
    get_image_scale,
    get_matches,
    group_by_file_name,
)
from .base_builders import ViewConfBuilder

BASE_IMAGE_VIEW_TYPE = "image"
SEG_IMAGE_VIEW_TYPE = "seg"
KAGGLE_IMAGE_VIEW_TYPE = "kaggle-seg"
GEOMX_IMAGE_VIEW_TYPE = "geomx-seg"


class AbstractImagingViewConfBuilder(ViewConfBuilder):
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        self.image_pyramid_regex = None
        self.seg_image_pyramid_regex = None
        self.use_full_resolution = []
        self.use_physical_size_scaling = False
        self.view_type = BASE_IMAGE_VIEW_TYPE
        self.base_image_metadata = None
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)

    def _get_img_and_offset_url(self, img_path, img_dir):
        """Create a url for the offsets and img.
        :param str img_path: The path of the image
        :param str img_dir: The image-specific part of the path to be
        replaced by the OFFSETS_DIR constant.
        :rtype: tuple The image url and the offsets url

        >>> from pprint import pprint
        >>> class ConcreteBuilder(AbstractImagingViewConfBuilder):
        ...     def get_conf_cells(self, **kwargs):
        ...         pass
        >>> builder = ConcreteBuilder(
        ...   entity={ "uuid": "uuid" },
        ...   groups_token='groups_token',
        ...   assets_endpoint='https://example.com')
        >>> pprint(builder._get_img_and_offset_url("rel_path/to/clusters.ome.tiff", "rel_path/to"))
        ('https://example.com/uuid/rel_path/to/clusters.ome.tiff?token=groups_token',\n\
         'https://example.com/uuid/output_offsets/clusters.offsets.json?token=groups_token',\n\
         'https://example.com/uuid/image_metadata/clusters.metadata.json?token=groups_token')

        """
        img_url = self._build_assets_url(img_path)
        return (
            img_url,
            str(
                re.sub(
                    r"ome\.tiff?",
                    "offsets.json",
                    re.sub(img_dir, OFFSETS_DIR, img_url),
                )
            ),
            str(
                re.sub(
                    r"ome\.tiff?",
                    "metadata.json",
                    re.sub(img_dir, IMAGE_METADATA_DIR, img_url),
                )
            ),
        )

    def _get_img_and_offset_url_seg(self, img_path, img_dir):
        """Create a url for the offsets and img for the EPICs base-image support datasets.
        :param str img_path: The path of the image
        :param str img_dir: The image-specific part of the path to be
        replaced by the OFFSETS_DIR constant.
        :rtype: tuple The image url and the offsets url

        """
        img_url = self._build_assets_url(img_path)
        offsets_path = re.sub(IMAGE_PYRAMID_DIR, OFFSETS_DIR, img_dir)
        metadata_path = re.sub(IMAGE_PYRAMID_DIR, IMAGE_METADATA_DIR, img_dir)
        return (
            img_url,
            str(
                re.sub(
                    r"ome\.tiff?",
                    "offsets.json",
                    re.sub(img_dir, offsets_path, img_url),
                )
            ),
            str(
                re.sub(
                    r"ome\.tiff?",
                    "metadata.json",
                    re.sub(img_dir, metadata_path, img_url),
                )
            ),
        )

    def _add_segmentation_image(self, dataset):
        file_paths_found = self._get_file_paths()

        if any(".zarr.zip" in path for path in file_paths_found):
            self._is_zarr_zip = True
        if self.seg_image_pyramid_regex is None:
            raise ValueError("seg_image_pyramid_regex is not set. Cannot find segmentation images.")

        found_images = get_found_images(self.seg_image_pyramid_regex, file_paths_found)

        filtered_images = [img for img in found_images if not any(subdir in img for subdir in base_image_dirs)]

        if not filtered_images:
            raise FileNotFoundError(f"Segmentation assay with uuid {self._uuid} has no matching files")

        img_url, offsets_url, metadata_url = self._get_img_and_offset_url(
            filtered_images[0], self.seg_image_pyramid_regex
        )
        seg_meta_data = get_image_metadata(self, metadata_url)

        scale = get_image_scale(self.base_image_metadata, seg_meta_data)
        if dataset is not None:
            if self.view_type == GEOMX_IMAGE_VIEW_TYPE:
                dataset.add_object(
                    ObsSegmentationsOmeTiffWrapper(
                        img_url=img_url,
                        offsets_url=offsets_url,
                        coordination_values={"fileUid": "segmentations"},
                        obs_types_from_channel_names=True,
                        coordinate_transformations=[{"type": "scale", "scale": scale}],
                    )
                )
            else:
                dataset.add_object(
                    ObsSegmentationsOmeTiffWrapper(
                        img_url=img_url,
                        offsets_url=offsets_url,
                        obs_types_from_channel_names=True,
                        coordinate_transformations=[{"type": "scale", "scale": scale}],
                    )
                )

    def _get_url_for_path(self, base, file_name, zip_check=False):
        file_paths_found = self._get_file_paths()
        file_name_end = f"{file_name}.zip" if self._is_zarr_zip and zip_check else file_name
        file_name_to_check = (
            f"{file_name_end}/.zgroup" if (".zarr" in file_name_end and not self._is_zarr_zip) else file_name_end
        )

        found_file = next(
            (p for p in file_paths_found if p.startswith(f"{base.rstrip('/')}/") and p.endswith(file_name_to_check)),
            None,
        )
        if found_file:
            if found_file.endswith("/.zgroup"):
                found_file = found_file[: -len("/.zgroup")]
            return self._build_assets_url(found_file)
        else:  # pragma: no cover
            print(f"{file_name_to_check} file was not found.")
            return None

    def _add_aoi_rois(self, dataset):
        area_zarr_url = self._get_url_for_path(self.segment_files_regex, "aoi.zarr", zip_check=True)
        area_zarr = AnnDataWrapper(
            adata_url=area_zarr_url,
            is_zip=self._is_zarr_zip,
            obs_set_paths=["obs/roi_id"],
            obs_set_names=["ROI"],
            coordination_values={"obsType": "area"},
        )

        region_zarr_url = self._get_url_for_path(self.segment_files_regex, "roi.zarr", zip_check=True)
        region_zarr = AnnDataWrapper(
            adata_url=region_zarr_url,
            is_zip=self._is_zarr_zip,
            coordination_values={"obsType": "region"},
        )
        dataset.add_object(region_zarr)
        dataset.add_object(area_zarr)

    def _setup_view_config(self, vc, dataset, view_type, disable_3d=[], use_full_resolution=[]):
        if view_type == BASE_IMAGE_VIEW_TYPE:
            vc.add_view(cm.SPATIAL, dataset=dataset, x=3, y=0, w=9, h=12).set_props(
                useFullResolutionImage=use_full_resolution
            )
            vc.add_view(cm.DESCRIPTION, dataset=dataset, x=0, y=8, w=3, h=4)
            vc.add_view(cm.LAYER_CONTROLLER, dataset=dataset, x=0, y=0, w=3, h=8).set_props(
                disable3d=disable_3d, disableChannelsIfRgbDetected=True
            )
        if view_type == GEOMX_IMAGE_VIEW_TYPE:
            self._add_views(vc, dataset)

        elif "seg" in view_type:
            spatial_view = vc.add_view("spatialBeta", dataset=dataset, x=4, y=0, w=8, h=12).set_props(
                useFullResolutionImage=use_full_resolution
            )
            lc_view = vc.add_view("layerControllerBeta", dataset=dataset, x=0, y=0, w=4, h=8).set_props(
                disable3d=disable_3d, disableChannelsIfRgbDetected=True
            )
            # Adding the segmentation mask on top of the image
            if view_type == KAGGLE_IMAGE_VIEW_TYPE:
                # vc.link_views_by_dict([spatial_view, lc_view])
                # TODO: The image-channel view disappears after the following
                vc.link_views_by_dict(
                    [spatial_view, lc_view],
                    {
                        "imageLayer": CL(
                            [
                                {
                                    "photometricInterpretation": "RGB",
                                }
                            ]
                        ),
                    },
                    meta=True,
                    scope_prefix=get_initial_coordination_scope_prefix("A", "image"),
                )

        return vc

    def _add_views(self, vc, dataset):
        spatial_view = vc.add_view("spatialBeta", dataset=dataset, w=8, h=12)
        lc_view = vc.add_view("layerControllerBeta", dataset=dataset, w=4, h=12, x=8, y=0)

        vc.link_views_by_dict(
            [spatial_view, lc_view],
            {
                "spatialTargetZ": 0,
                "spatialTargetT": 0,
                "imageLayer": CL(
                    [
                        {
                            "fileUid": "image",
                            "photometricInterpretation": "BlackIsZero",
                        }
                    ]
                ),
            },
            meta=True,
            scope_prefix=get_initial_coordination_scope_prefix("A", "image"),
        )
        vc.link_views_by_dict(
            [spatial_view, lc_view],
            {
                "segmentationLayer": CL(
                    [
                        {
                            "fileUid": "segmentations",
                            "spatialLayerOpacity": 1.0,
                            "spatialLayerVisible": True,
                            "segmentationChannel": CL(
                                [
                                    {
                                        "spatialTargetC": 0,
                                        "obsType": "Full ROI",
                                        "spatialChannelColor": [155, 165, 31],
                                        "spatialChannelOpacity": 0.8,
                                        "obsHighlight": None,
                                        "spatialChannelVisible": True,
                                        "obsColorEncoding": "spatialChannelColor",
                                        "spatialSegmentationFilled": True,
                                        "spatialSegmentationStrokeWidth": 0.01,
                                    },
                                ]
                            ),
                        }
                    ]
                ),
            },
            meta=True,
            scope_prefix=get_initial_coordination_scope_prefix("A", "obsSegmentations"),
        )

    def get_conf_cells_common(self, get_img_and_offset_url_func, **kwargs):
        file_paths_found = self._get_file_paths()
        found_images = get_found_images(self.image_pyramid_regex, file_paths_found)
        found_images = sorted(found_images)
        if len(found_images) == 0:  # pragma: no cover
            message = f"Image pyramid assay with uuid {self._uuid} has no matching files"
            raise FileNotFoundError(message)

        vc = VitessceConfig(name="HuBMAP Data Portal", schema_version=self._schema_version)
        dataset = vc.add_dataset(name="Visualization Files")

        if "seg" in self.view_type:
            img_url, offsets_url, metadata_url = get_img_and_offset_url_func(found_images[0], self.image_pyramid_regex)
            meta_data = get_image_metadata(self, metadata_url)
            self.base_image_metadata = meta_data
            if self.view_type == GEOMX_IMAGE_VIEW_TYPE:
                dataset = dataset.add_object(
                    ImageOmeTiffWrapper(
                        img_url=img_url,
                        offsets_url=offsets_url,
                        name=Path(found_images[0]).name,
                        coordination_values={"fileUid": "image"},
                    )
                )
            else:
                dataset = dataset.add_object(
                    ImageOmeTiffWrapper(
                        img_url=img_url,
                        offsets_url=offsets_url,
                        name=Path(found_images[0]).name,
                    )
                )
            if self.view_type in [KAGGLE_IMAGE_VIEW_TYPE, GEOMX_IMAGE_VIEW_TYPE]:
                self._add_segmentation_image(dataset)

        else:
            images = [
                OmeTiffWrapper(
                    img_url=img_url,
                    offsets_url=offsets_url,
                    name=Path(img_path).name,
                )
                for img_path in found_images
                for img_url, offsets_url, _ in [get_img_and_offset_url_func(img_path, self.image_pyramid_regex)]
            ]
            dataset.add_object(MultiImageWrapper(images, use_physical_size_scaling=self.use_physical_size_scaling))

        if self.view_type == GEOMX_IMAGE_VIEW_TYPE:
            self._add_aoi_rois(dataset)
        conf = self._setup_view_config(
            vc, dataset, self.view_type, use_full_resolution=self.use_full_resolution
        ).to_dict()
        if self.view_type == BASE_IMAGE_VIEW_TYPE:
            del conf["datasets"][0]["files"][0]["options"]["renderLayers"]
        return get_conf_cells(conf)


class ImagePyramidViewConfBuilder(AbstractImagingViewConfBuilder):
    """Wrapper class for creating a standard view configuration for image pyramids,
    i.e for high resolution viz-lifted imaging datasets like
    https://portal.hubmapconsortium.org/browse/dataset/dc289471333309925e46ceb9bafafaf4
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self.image_pyramid_regex = IMAGE_PYRAMID_DIR
        self.view_type = BASE_IMAGE_VIEW_TYPE

    def get_conf_cells(self, **kwargs):
        return self.get_conf_cells_common(self._get_img_and_offset_url, **kwargs)


class EpicSegImagePyramidViewConfBuilder(AbstractImagingViewConfBuilder):
    """Wrapper class for creating a standard view configuration for image pyramids for EPIC segmentation mask,
    i.e for high resolution viz-lifted imaging datasets like
    https://portal.dev.hubmapconsortium.org/browse/dataset/df7cac7cb67a822f7007b57c4d8f5e7d
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self.image_pyramid_regex = f"{SEGMENTATION_SUBDIR}/{IMAGE_PYRAMID_DIR}/{SEGMENTATION_SUPPORT_IMAGE_SUBDIR}"
        self.view_type = SEG_IMAGE_VIEW_TYPE

    def get_conf_cells(self, **kwargs):
        return self.get_conf_cells_common(self._get_img_and_offset_url_seg, **kwargs)


class KaggleSegImagePyramidViewConfBuilder(AbstractImagingViewConfBuilder):
    """Wrapper class for creating a standard view configuration for image pyramids for kaggle-2 datasets, that show,
    segmentation mask layered over a base image-pyramid, however, the file structure is different than
    EPIC segmentation masks (EpicSegImagePyramidViewConfBuilder)
    i.e for high resolution viz-lifted imaging datasets like
    https://portal.dev.hubmapconsortium.org/browse/dataset/534a590d7336aa99c7fc7afd41e995fc
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self.seg_image_pyramid_regex = IMAGE_PYRAMID_DIR
        self.view_type = KAGGLE_IMAGE_VIEW_TYPE

        # Needed to adjust to various directory structures. For older datasets, the image pyramids will be present in
        # 'processed_microscopy' or 'processedMicroscopy' while newer datasets are listed under lab_processed.

        image_dir = SEGMENTATION_SUPPORT_IMAGE_SUBDIR
        file_paths_found = self._get_file_paths()
        paths = get_found_images_all(file_paths_found)
        matched_dirs = {dir for dir in base_image_dirs if any(dir in img for img in paths)}

        image_dir = next(iter(matched_dirs), image_dir)

        self.image_pyramid_regex = f"{IMAGE_PYRAMID_DIR}/{image_dir}"

    def get_conf_cells(self, **kwargs):
        return self.get_conf_cells_common(self._get_img_and_offset_url_seg, **kwargs)


class GeoMxImagePyramidViewConfBuilder(AbstractImagingViewConfBuilder):
    """Wrapper class for creating a view configuration for image pyramids for GeoMx datasets, that show,
    segmentation mask layered over a base image-pyramid with AOIs and ROIs highlighted.
    i.e for high resolution viz-lifted imaging datasets like
    TODO: add example https://portal.dev.hubmapconsortium.org/browse/dataset/
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self.seg_image_pyramid_regex = IMAGE_PYRAMID_DIR
        self.view_type = GEOMX_IMAGE_VIEW_TYPE
        self.segment_files_regex = GEOMX_DIR
        self._is_zarr_zip = False
        # file_paths_found = self._get_file_paths()
        # paths = get_found_images_all(file_paths_found)
        # print("path", paths)
        # matched_dirs = {dir for dir in base_image_dirs if any(dir in img for img in paths)}

        # image_dir = next(iter(matched_dirs), image_dir)
        # print("image", image_dir)

        self.image_pyramid_regex = f"{IMAGE_PYRAMID_DIR}/{SEGMENTATION_SUPPORT_IMAGE_SUBDIR}"

    def get_conf_cells(self, **kwargs):
        return self.get_conf_cells_common(self._get_img_and_offset_url_seg, **kwargs)


class IMSViewConfBuilder(ImagePyramidViewConfBuilder):
    """Wrapper class for generating a Vitessce configurations
    for IMS data that excludes the image pyramids
    of all the channels separated out.
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        # Do not show the separated mass-spec images.
        self.image_pyramid_regex = re.escape(IMAGE_PYRAMID_DIR) + r"(?!/ometiffs/separate/)"


class NanoDESIViewConfBuilder(ImagePyramidViewConfBuilder):
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        # Do not show full pyramid - does not look good
        image_names = [
            Path(file["rel_path"]).name for file in self._entity["files"] if not file["rel_path"].endswith("json")
        ]
        self.use_full_resolution = image_names
        self.use_physical_size_scaling = True


class SeqFISHViewConfBuilder(AbstractImagingViewConfBuilder):
    """Wrapper class for generating Vitessce configurations,
    one per position, with the hybridization cycles
    grouped together per position in a single Vitessce configuration.
    """

    def get_conf_cells(self, **kwargs):
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]
        full_seqfish_regex = "/".join([IMAGE_PYRAMID_DIR, SEQFISH_HYB_CYCLE_REGEX, SEQFISH_FILE_REGEX])
        found_images = get_matches(file_paths_found, full_seqfish_regex)
        if len(found_images) == 0:
            message = f"seqFish assay with uuid {self._uuid} has no matching files"
            raise FileNotFoundError(message)
        # Get all files grouped by PosN names.
        images_by_pos = group_by_file_name(found_images)
        confs = []
        # Build up a conf for each Pos.
        for images in images_by_pos:
            image_wrappers = []
            pos_name = self._get_pos_name(images[0])
            vc = VitessceConfig(name=pos_name, schema_version=self._schema_version)
            dataset = vc.add_dataset(name=pos_name)
            sorted_images = sorted(images, key=self._get_hybcycle)
            for img_path in sorted_images:
                img_url, offsets_url, _ = self._get_img_and_offset_url(img_path, IMAGE_PYRAMID_DIR)
                image_wrappers.append(
                    OmeTiffWrapper(
                        img_url=img_url,
                        offsets_url=offsets_url,
                        name=self._get_hybcycle(img_path),
                    )
                )
            dataset = dataset.add_object(MultiImageWrapper(image_wrappers))
            vc = self._setup_view_config(
                vc, dataset, self.view_type, disable_3d=[self._get_hybcycle(img_path) for img_path in sorted_images]
            )
            conf = vc.to_dict()
            # Don't want to render all layers
            del conf["datasets"][0]["files"][0]["options"]["renderLayers"]
            confs.append(conf)
        return get_conf_cells(confs)

    def _get_hybcycle(self, image_path):
        return re.search(SEQFISH_HYB_CYCLE_REGEX, image_path)[0]

    def _get_pos_name(self, image_path):
        return re.search(SEQFISH_FILE_REGEX, image_path)[0].split(".")[0]
