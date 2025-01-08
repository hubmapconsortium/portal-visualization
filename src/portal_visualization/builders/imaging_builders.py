from pathlib import Path
import re

from vitessce import (
    VitessceConfig,
    MultiImageWrapper,
    OmeTiffWrapper,
    CoordinationLevel as CL,
    get_initial_coordination_scope_prefix,
    ObsSegmentationsOmeTiffWrapper,
    ImageOmeTiffWrapper,
    Component as cm,
)

from ..utils import get_matches, group_by_file_name, get_conf_cells
from ..paths import (IMAGE_PYRAMID_DIR, OFFSETS_DIR, SEQFISH_HYB_CYCLE_REGEX,
                     SEQFISH_FILE_REGEX, SEGMENTATION_SUPPORT_IMAGE_SUBDIR,
                     SEGMENTATION_SUBDIR)
from .base_builders import ViewConfBuilder


class AbstractImagingViewConfBuilder(ViewConfBuilder):
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        self.image_pyramid_regex = None
        self.seg_image_pyramid_regex = None
        self.use_full_resolution = []
        self.use_physical_size_scaling = False
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
        offsets_path= re.sub(IMAGE_PYRAMID_DIR, OFFSETS_DIR, img_dir)
        return (
            img_url,
            str(
                re.sub(
                    r"ome\.tiff?",
                    "offsets.json",
                    re.sub(img_dir, offsets_path, img_url),
                )
            ),
        )
    
    def _add_segmentation_image(self, dataset):
            file_paths_found = self._get_file_paths()
            found_images = get_found_images(self.seg_image_pyramid_regex, file_paths_found)
            filtered_images = [img for img in found_images if SEGMENTATION_SUPPORT_IMAGE_SUBDIR not in img]

            if not filtered_images:
                raise FileNotFoundError(f"Segmentation assay with uuid {self._uuid} has no matching files")

            img_url, offsets_url = self._get_img_and_offset_url(filtered_images[0], self.seg_image_pyramid_regex)
            dataset.add_object(
                ObsSegmentationsOmeTiffWrapper(img_url=img_url, offsets_url=offsets_url, obs_types_from_channel_names=True,
                # coordinate_transformations=[{"type": "scale", "scale": [0.377.,0.377,1,1,1]}] # need to read from a file
                )
            )

    def _setup_view_config(self, vc, dataset, view_type, disable_3d=[], use_full_resolution=[]):
        if view_type == "image":
            vc.add_view(cm.SPATIAL, dataset=dataset, x=3, y=0, w=9, h=12).set_props(
                useFullResolutionImage=use_full_resolution
            )
            vc.add_view(cm.DESCRIPTION, dataset=dataset, x=0, y=8, w=3, h=4)
            vc.add_view(cm.LAYER_CONTROLLER, dataset=dataset, x=0, y=0, w=3, h=8).set_props(
                disable3d=disable_3d, disableChannelsIfRgbDetected=True
            )
        elif "seg" in view_type:
            spatial_view = vc.add_view("spatialBeta", dataset=dataset, x=4, y=0, w=8, h=12).set_props(
                useFullResolutionImage=use_full_resolution
            )
            lc_view = vc.add_view("layerControllerBeta", dataset=dataset, x=0, y=0, w=4, h=8).set_props(
                disable3d=disable_3d, disableChannelsIfRgbDetected=True
            )
            # Adding the segmentation mask on top of the image
            if view_type == 'kaggle_seg':
                vc.link_views_by_dict([spatial_view, lc_view], {
                'imageLayer': CL([{'photometricInterpretation': 'RGB', }]),
            }, meta=True, scope_prefix=get_initial_coordination_scope_prefix("A", "image"))

        return vc

    def get_conf_cells_common(self, get_img_and_offset_url_func, **kwargs):
        file_paths_found = self._get_file_paths()
        found_images = get_found_images(self.image_pyramid_regex, file_paths_found)
        found_images = sorted(found_images)
        if len(found_images) == 0:  # pragma: no cover
            message = f"Image pyramid assay with uuid {self._uuid} has no matching files"
            raise FileNotFoundError(message)

        vc = VitessceConfig(name="HuBMAP Data Portal", schema_version=self._schema_version)
        dataset = vc.add_dataset(name="Visualization Files")

        if 'seg' in self.view_type:
            img_url, offsets_url = get_img_and_offset_url_func(found_images[0], self.image_pyramid_regex)
            dataset = dataset.add_object(
                ImageOmeTiffWrapper(img_url=img_url, offsets_url=offsets_url, name=Path(found_images[0]).name)
            )
            if self.view_type == 'kaggle-seg':
                self._add_segmentation_image(dataset)
                
                
        else:
            images = [
                OmeTiffWrapper(
                    img_url=img_url, offsets_url=offsets_url, name=Path(img_path).name
                )
                for img_path in found_images
                for img_url, offsets_url in [get_img_and_offset_url_func(img_path, self.image_pyramid_regex)]
            ]
            dataset.add_object(
                MultiImageWrapper(images, use_physical_size_scaling=self.use_physical_size_scaling)
            )
        conf = self._setup_view_config(vc, dataset, self.view_type, use_full_resolution=self.use_full_resolution).to_dict()
        if "image" in self.view_type:
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
        self.view_type = "image"

    def get_conf_cells(self, **kwargs):
        return self.get_conf_cells_common(self._get_img_and_offset_url, **kwargs)


class SegImagePyramidViewConfBuilder(AbstractImagingViewConfBuilder):
    """Wrapper class for creating a standard view configuration for image pyramids for segmenation mask,
    i.e for high resolution viz-lifted imaging datasets like
    https://portal.hubmapconsortium.org/browse/dataset/
    """
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self.image_pyramid_regex = f"{SEGMENTATION_SUBDIR}/{IMAGE_PYRAMID_DIR}/{SEGMENTATION_SUPPORT_IMAGE_SUBDIR}"
        self.view_type = "seg"

    def get_conf_cells(self, **kwargs):
        return self.get_conf_cells_common(self._get_img_and_offset_url_seg, **kwargs)
  
class KaggleSegImagePyramidViewConfBuilder(AbstractImagingViewConfBuilder):
    # The difference from EPIC segmentation is only the file path and transformations
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self.image_pyramid_regex = f"{IMAGE_PYRAMID_DIR}/{SEGMENTATION_SUPPORT_IMAGE_SUBDIR}"
        self.seg_image_pyramid_regex = IMAGE_PYRAMID_DIR
        self.view_type = "kaggle-seg"

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
    
def get_found_images(image_pyramid_regex, file_paths_found):
    found_images = [
        path for path in get_matches(
            file_paths_found, image_pyramid_regex + r".*\.ome\.tiff?$",
        )
        if 'separate/' not in path
    ]
    return found_images