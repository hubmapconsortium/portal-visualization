from pathlib import Path
import re

from vitessce import (
    VitessceConfig,
    MultiImageWrapper,
    OmeTiffWrapper,
    ImageOmeTiffWrapper,
    Component as cm,
)

from ..utils import get_matches, group_by_file_name, get_conf_cells
from ..paths import (IMAGE_PYRAMID_DIR, OFFSETS_DIR, SEQFISH_HYB_CYCLE_REGEX,
                     SEQFISH_FILE_REGEX, SEGMENTATION_SUPPORT_IMAGE_SUBDIR)
from .base_builders import ViewConfBuilder


class AbstractImagingViewConfBuilder(ViewConfBuilder):
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
         'https://example.com/uuid/output_offsets/clusters.offsets.json?token=groups_token')

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
        )

    def _get_img_and_offset_url_seg(self, img_path, img_dir):
        """Create a url for the offsets and img for the EPICs base-image support datasets.
        :param str img_path: The path of the image
        :param str img_dir: The image-specific part of the path to be
        replaced by the OFFSETS_DIR constant.
        :rtype: tuple The image url and the offsets url

        """
        img_url = self._build_assets_url(img_path)
        offset_path = f'{OFFSETS_DIR}/{SEGMENTATION_SUPPORT_IMAGE_SUBDIR}'
        return (
            img_url,
            str(
                re.sub(
                    r"ome\.tiff?",
                    "offsets.json",
                    re.sub(img_dir, offset_path, img_url),
                )
            ),
        )

    def _setup_view_config_raster(self, vc, dataset, disable_3d=[], use_full_resolution=[]):
        vc.add_view(cm.SPATIAL, dataset=dataset, x=3, y=0, w=9, h=12).set_props(
            useFullResolutionImage=use_full_resolution
        )
        vc.add_view(cm.DESCRIPTION, dataset=dataset, x=0, y=8, w=3, h=4)
        vc.add_view(cm.LAYER_CONTROLLER, dataset=dataset, x=0, y=0, w=3, h=8).set_props(
            disable3d=disable_3d, disableChannelsIfRgbDetected=True
        )
        return vc

    def _setup_view_config_seg(self, vc, dataset, disable_3d=[], use_full_resolution=[]):
        vc.add_view("spatialBeta", dataset=dataset, x=3, y=0, w=9, h=12).set_props(
            useFullResolutionImage=use_full_resolution
        )
        vc.add_view("layerControllerBeta", dataset=dataset, x=0, y=0, w=3, h=8).set_props(
            disable3d=disable_3d, disableChannelsIfRgbDetected=True
        )
        return vc


class ImagePyramidViewConfBuilder(AbstractImagingViewConfBuilder):
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        """Wrapper class for creating a standard view configuration for image pyramids,
        i.e for high resolution viz-lifted imaging datasets like
        https://portal.hubmapconsortium.org/browse/dataset/dc289471333309925e46ceb9bafafaf4
        """
        self.image_pyramid_regex = IMAGE_PYRAMID_DIR
        self.use_full_resolution = []
        self.use_physical_size_scaling = False
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)

    def get_conf_cells(self, **kwargs):
        file_paths_found = self._get_file_paths()
        found_images = [
            path for path in get_matches(
                file_paths_found, self.image_pyramid_regex + r".*\.ome\.tiff?$",
            )
            if 'separate/' not in path  # Exclude separate/* in MALDI-IMS
        ]
        found_images = sorted(found_images)
        if len(found_images) == 0:
            message = f"Image pyramid assay with uuid {self._uuid} has no matching files"
            raise FileNotFoundError(message)

        vc = VitessceConfig(name="HuBMAP Data Portal", schema_version=self._schema_version)
        dataset = vc.add_dataset(name="Visualization Files")
        images = []
        for img_path in found_images:
            img_url, offsets_url = self._get_img_and_offset_url(
                img_path, self.image_pyramid_regex
            )
            images.append(
                OmeTiffWrapper(
                    img_url=img_url, offsets_url=offsets_url, name=Path(img_path).name
                )
            )
        dataset = dataset.add_object(
            MultiImageWrapper(
                images,
                use_physical_size_scaling=self.use_physical_size_scaling
            )
        )
        vc = self._setup_view_config_raster(
            vc, dataset, use_full_resolution=self.use_full_resolution)
        conf = vc.to_dict()
        # Don't want to render all layers
        del conf["datasets"][0]["files"][0]["options"]["renderLayers"]
        return get_conf_cells(conf)


class SegImagePyramidViewConfBuilder(AbstractImagingViewConfBuilder):
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        """Wrapper class for creating a standard view configuration for image pyramids for segmenation mask,
        i.e for high resolution viz-lifted imaging datasets like
        https://portal.hubmapconsortium.org/browse/dataset/
        """
        self.image_pyramid_regex = f'{IMAGE_PYRAMID_DIR}/{SEGMENTATION_SUPPORT_IMAGE_SUBDIR}'
        self.use_full_resolution = []
        self.use_physical_size_scaling = False
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)

    def get_conf_cells(self, **kwargs):
        file_paths_found = self._get_file_paths()
        found_images = [
            path for path in get_matches(
                file_paths_found, self.image_pyramid_regex + r".*\.ome\.tiff?$",
            )
            if 'separate/' not in path  # Exclude separate/* in MALDI-IMS
        ]
        found_images = sorted(found_images)
        if len(found_images) == 0:
            message = f"Image pyramid assay with uuid {self._uuid} has no matching files"
            raise FileNotFoundError(message)

        vc = VitessceConfig(name="HuBMAP Data Portal", schema_version=self._schema_version)
        dataset = vc.add_dataset(name="Visualization Files")
        # The base-image will always be 1
        if len(found_images) == 1:
            img_url, offsets_url = self._get_img_and_offset_url_seg(
                found_images[0], self.image_pyramid_regex
            )

            image = ImageOmeTiffWrapper(
                img_url=img_url, offsets_url=offsets_url, name=Path(found_images[0]).name
            )

            dataset = dataset.add_object(image)
        vc = self._setup_view_config_seg(
            vc, dataset, use_full_resolution=self.use_full_resolution)
        conf = vc.to_dict()
        return get_conf_cells(conf)


class IMSViewConfBuilder(ImagePyramidViewConfBuilder):
    """Wrapper class for generating a Vitessce configurations
    for IMS data that excludes the image pyramids
    of all the channels separated out.
    """

    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        # Do not show the separated mass-spec images.
        self.image_pyramid_regex = (
            re.escape(IMAGE_PYRAMID_DIR) + r"(?!/ometiffs/separate/)"
        )


class NanoDESIViewConfBuilder(ImagePyramidViewConfBuilder):
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        # Do not show full pyramid - does not look good
        image_names = [Path(file['rel_path']).name for file in self._entity["files"]
                       if not file["rel_path"].endswith('json')]
        self.use_full_resolution = image_names
        self.use_physical_size_scaling = True


class SeqFISHViewConfBuilder(AbstractImagingViewConfBuilder):
    """Wrapper class for generating Vitessce configurations,
    one per position, with the hybridization cycles
    grouped together per position in a single Vitessce configuration.
    """

    def get_conf_cells(self, **kwargs):
        file_paths_found = [file["rel_path"] for file in self._entity["files"]]
        full_seqfish_regex = "/".join(
            [
                IMAGE_PYRAMID_DIR,
                SEQFISH_HYB_CYCLE_REGEX,
                SEQFISH_FILE_REGEX
            ]
        )
        found_images = get_matches(file_paths_found, full_seqfish_regex)
        if len(found_images) == 0:
            message = f'seqFish assay with uuid {self._uuid} has no matching files'
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
                img_url, offsets_url = self._get_img_and_offset_url(
                    img_path, IMAGE_PYRAMID_DIR
                )
                image_wrappers.append(
                    OmeTiffWrapper(
                        img_url=img_url,
                        offsets_url=offsets_url,
                        name=self._get_hybcycle(img_path),
                    )
                )
            dataset = dataset.add_object(MultiImageWrapper(image_wrappers))
            vc = self._setup_view_config_raster(
                vc,
                dataset,
                disable_3d=[self._get_hybcycle(img_path) for img_path in sorted_images]
            )
            conf = vc.to_dict()
            # Don't want to render all layers
            del conf["datasets"][0]["files"][0]["options"]["renderLayers"]
            confs.append(conf)
        return get_conf_cells(confs)

    def _get_hybcycle(self, image_path):
        return re.search(SEQFISH_HYB_CYCLE_REGEX, image_path)[0]

    def _get_pos_name(self, image_path):
        return re.search(SEQFISH_FILE_REGEX, image_path)[0].split(".")[
            0
        ]
