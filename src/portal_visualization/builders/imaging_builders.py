import io
import json
import logging
import re
import zipfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
from vitessce import (
    AnnDataWrapper,
    ImageOmeTiffWrapper,
    ObsSegmentationsOmeTiffWrapper,
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
    get_ome_tiff_metadata,
    group_by_file_name,
    with_config_builder_user_agent,
)
from .base_builders import ViewConfBuilder

logger = logging.getLogger(__name__)

BASE_IMAGE_VIEW_TYPE = "image"
KAGGLE_IMAGE_VIEW_TYPE = "kaggle-seg"
GEOMX_IMAGE_VIEW_TYPE = "geomx-seg"

# Deterministic color palette for segmentation channels.
SEGMENTATION_CHANNEL_COLORS = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [44, 160, 44],
    [148, 103, 189],
    [255, 127, 14],
]


class AbstractImagingViewConfBuilder(ViewConfBuilder):
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self.image_pyramid_regex = None
        self.seg_image_pyramid_regex = None
        self.use_full_resolution = []
        self.view_type = BASE_IMAGE_VIEW_TYPE
        self.base_image_metadata = None
        self.seg_channel_count = 1
        self.seg_channel_names = None

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

        # The segmentation pyramid historically lives outside the base-image dirs, so exclude those.
        # Newer pipelines emit it under ``lab_processed/images/`` next to the base image, where that
        # exclusion leaves nothing — fall back to the ``*.segmentations.*`` suffix in that case.
        filtered_images = [img for img in found_images if not any(subdir in img for subdir in base_image_dirs)]
        if not filtered_images:
            filtered_images = [img for img in found_images if ".segmentations." in img]

        if not filtered_images:
            raise FileNotFoundError(f"Dataset {self._uuid} is missing segmentation image pyramid files")

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
            logger.warning("%s file was not found.", file_name_to_check)
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

    def _get_seg_channel_info(self):  # pragma: no cover
        """Fetch segmentation channel names and count from the AOI zarr's segment categories.

        Reads obs/segment/categories/.zarray (metadata) and obs/segment/categories/0 (data)
        from the AOI zarr, then decodes the binary chunk to get channel name strings.
        Falls back to count-only from shape[0], or to 1 on any error.
        """
        from numcodecs import Blosc, VLenUTF8

        try:
            area_zarr_url = self._get_url_for_path(self.segment_files_regex, "aoi.zarr", zip_check=True)
            if not area_zarr_url:
                return

            request_init = with_config_builder_user_agent(self._get_request_init())

            if self._is_zarr_zip:
                response = requests.get(area_zarr_url, **request_init)
                if response.status_code != 200:
                    logger.warning("Failed to fetch AOI zarr zip: %s", response.status_code)
                    return
                zf = zipfile.ZipFile(io.BytesIO(response.content))
                zarray_data = json.loads(zf.read("obs/segment/categories/.zarray"))
                chunk_bytes = zf.read("obs/segment/categories/0")
            else:
                parsed = urlparse(area_zarr_url)

                zarray_path = f"{parsed.path}/obs/segment/categories/.zarray"
                zarray_url = urlunparse(parsed._replace(path=zarray_path))
                response = requests.get(zarray_url, **request_init)
                if response.status_code != 200:
                    logger.warning("Failed to fetch segment categories .zarray: %s", response.status_code)
                    return
                zarray_data = response.json()

                chunk_path = f"{parsed.path}/obs/segment/categories/0"
                chunk_url = urlunparse(parsed._replace(path=chunk_path))
                chunk_response = requests.get(chunk_url, **request_init)
                if chunk_response.status_code != 200:
                    logger.warning("Failed to fetch segment categories chunk: %s", chunk_response.status_code)
                    self.seg_channel_count = zarray_data["shape"][0]
                    return
                chunk_bytes = chunk_response.content

            # Decode the binary chunk using compressor and filter from .zarray metadata
            compressor_config = zarray_data.get("compressor")
            if compressor_config:
                decompressed = Blosc(**{k: v for k, v in compressor_config.items() if k != "id"}).decode(chunk_bytes)
            else:
                decompressed = chunk_bytes

            names = list(VLenUTF8().decode(decompressed))
            self.seg_channel_names = names
            self.seg_channel_count = len(names)
        except Exception as e:
            logger.warning("Could not determine segment channel info from AOI zarr: %s", e)

    def _build_segmentation_channel_entries(self):
        """Build segmentation channel coordination entries based on channel count."""
        entries = []
        for i in range(self.seg_channel_count):
            color = SEGMENTATION_CHANNEL_COLORS[i % len(SEGMENTATION_CHANNEL_COLORS)]
            entries.append(
                {
                    "spatialTargetC": i,
                    "obsType": self.seg_channel_names[i] if self.seg_channel_names else f"Segment {i}",
                    "spatialChannelColor": color,
                    "spatialChannelOpacity": 0.8,
                    "obsHighlight": None,
                    "spatialChannelVisible": True,
                    "obsColorEncoding": "spatialChannelColor",
                    "spatialSegmentationFilled": True,
                    "spatialSegmentationStrokeWidth": 0.01,
                }
            )
        return entries

    def _link_base_image_layers(self, vc, spatial_view, lc_view, image_fileuids, image_channel_count):
        """Wire the base-image layers into the beta spatial + layer-controller views.

        A single image just links both views to the sole (auto-initialized) layer. With multiple
        images the beta spatial view rasterizes EVERY loaded image, so each needs a fully-specified
        layer (an absent/partial channel list dereferences ``undefined`` and crashes the view) — but
        we don't want one controller panel per tile. Following Vitessce's "proxy layer" pattern, the
        layer controller is linked to a single proxy layer while the spatial view is linked to all
        tiles; both reference the SAME shared coordination scopes (visibility, opacity, photometric,
        and the ``imageChannel`` color/contrast-slider scopes), so the one controller panel drives
        every tile at once. The two ``link_views_by_dict`` calls share a scope prefix; the numbering
        skips already-used scope names, so they don't collide.
        """
        fileuids = image_fileuids or []
        prefix = get_initial_coordination_scope_prefix("A", "image")
        if len(fileuids) <= 1:
            vc.link_views_by_dict(
                [spatial_view, lc_view],
                {"imageLayer": CL([{"fileUid": uid} for uid in fileuids])},
                meta=True,
                scope_prefix=prefix,
            )
            return

        num_channels = max(1, min(image_channel_count or 1, len(SEGMENTATION_CHANNEL_COLORS)))
        visible, opacity, photometric = vc.add_coordination(
            "spatialLayerVisible", "spatialLayerOpacity", "photometricInterpretation"
        )
        visible.set_value(True)
        opacity.set_value(1.0)
        photometric.set_value("BlackIsZero")
        # Channel labels drawn once (on the proxy / first tile), not duplicated across every tile.
        labels_on, labels_off = vc.add_coordination("spatialChannelLabelsVisible", "spatialChannelLabelsVisible")
        labels_on.set_value(True)
        labels_off.set_value(False)
        # One CL instance -> its per-channel scopes (incl. the contrast slider) are shared by every layer.
        shared_channels = CL(
            [
                {
                    "spatialTargetC": c,
                    "spatialChannelColor": SEGMENTATION_CHANNEL_COLORS[c],
                    "spatialChannelVisible": True,
                    "spatialChannelWindow": None,
                }
                for c in range(num_channels)
            ]
        )

        def layer(uid, labels_scope):
            return {
                "fileUid": uid,
                "spatialLayerVisible": visible,
                "spatialLayerOpacity": opacity,
                "photometricInterpretation": photometric,
                "spatialChannelLabelsVisible": labels_scope,
                "imageChannel": shared_channels,
            }

        # Controller: a single proxy layer -> one panel of channel sliders.
        vc.link_views_by_dict(
            [lc_view], {"imageLayer": CL([layer(fileuids[0], labels_on)])}, meta=True, scope_prefix=prefix
        )
        # Spatial view: every tile; channel labels drawn on the first tile only.
        vc.link_views_by_dict(
            [spatial_view],
            {"imageLayer": CL([layer(uid, labels_on if i == 0 else labels_off) for i, uid in enumerate(fileuids)])},
            meta=True,
            scope_prefix=prefix,
        )

    def _link_z_and_t(self, vc, spatial_view, lc_view):
        """Put the spatial and layer-controller views on one ``spatialTargetZ``/``spatialTargetT`` scope.

        Both types are in Vitessce's ``AUTO_INDEPENDENT_COORDINATION_TYPES``, so under
        ``initStrategy: auto`` each view is handed its *own* scope and the controller's Z slider
        never moves the image. Only z-stacked images surface this (seqFISH is 11 slices deep);
        ``_add_views`` already links them for GeoMx.
        """
        target_z, target_t = vc.add_coordination("spatialTargetZ", "spatialTargetT")
        target_z.set_value(0)
        target_t.set_value(0)
        spatial_view.use_coordination(target_z, target_t)
        lc_view.use_coordination(target_z, target_t)

    def _setup_view_config(
        self,
        vc,
        dataset,
        view_type,
        disable_3d=False,
        use_full_resolution=[],
        image_fileuids=None,
        image_channel_count=None,
    ):
        if view_type == BASE_IMAGE_VIEW_TYPE:
            spatial_view = vc.add_view("spatialBeta", dataset=dataset, x=3, y=0, w=9, h=12).set_props(
                useFullResolutionImage=use_full_resolution
            )
            vc.add_view(cm.DESCRIPTION, dataset=dataset, x=0, y=8, w=3, h=4)
            lc_view = vc.add_view("layerControllerBeta", dataset=dataset, x=0, y=0, w=3, h=8).set_props(
                globalDisable3d=disable_3d, disableChannelsIfRgbDetected=True
            )
            self._link_z_and_t(vc, spatial_view, lc_view)
            self._link_base_image_layers(vc, spatial_view, lc_view, image_fileuids, image_channel_count)
        if view_type == GEOMX_IMAGE_VIEW_TYPE:
            self._add_views(vc, dataset)

        elif "seg" in view_type:
            spatial_view = vc.add_view("spatialBeta", dataset=dataset, x=4, y=0, w=8, h=12).set_props(
                useFullResolutionImage=use_full_resolution
            )
            lc_view = vc.add_view("layerControllerBeta", dataset=dataset, x=0, y=0, w=4, h=8).set_props(
                globalDisable3d=disable_3d, disableChannelsIfRgbDetected=True
            )
            self._link_z_and_t(vc, spatial_view, lc_view)
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
                            "segmentationChannel": CL(self._build_segmentation_channel_entries()),
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
            raise FileNotFoundError(
                f"Dataset {self._uuid} is missing image pyramid files matching {self.image_pyramid_regex}"
            )

        vc, dataset = self._create_vitessce_config(dataset_name="Visualization Files")

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
            image_fileuids = [f"image_{i}" for i in range(len(found_images))]
            first_image_url = None
            for fileuid, img_path in zip(image_fileuids, found_images, strict=True):
                img_url, offsets_url, _ = get_img_and_offset_url_func(img_path, self.image_pyramid_regex)
                first_image_url = first_image_url or img_url
                dataset.add_object(
                    ImageOmeTiffWrapper(
                        img_url=img_url,
                        offsets_url=offsets_url,
                        name=Path(img_path).name,
                        coordination_values={"fileUid": fileuid},
                    )
                )
            # Multiple images need explicit per-layer channels (see _build_base_image_layers); read the
            # channel count once from the first image (they share an acquisition, so channels match).
            image_channel_count = None
            if len(found_images) > 1:
                image_channel_count = (get_ome_tiff_metadata(first_image_url) or {}).get("SizeC")

        if self.view_type == GEOMX_IMAGE_VIEW_TYPE:
            self._add_aoi_rois(dataset)
            self._get_seg_channel_info()
        conf = self._setup_view_config(
            vc,
            dataset,
            self.view_type,
            use_full_resolution=self.use_full_resolution,
            image_fileuids=image_fileuids if self.view_type == BASE_IMAGE_VIEW_TYPE else None,
            image_channel_count=image_channel_count if self.view_type == BASE_IMAGE_VIEW_TYPE else None,
        ).to_dict()
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


class KaggleSegImagePyramidViewConfBuilder(AbstractImagingViewConfBuilder):
    """Wrapper class for creating a standard view configuration for image pyramids for kaggle-2 datasets, that show,
    segmentation mask layered over a base image-pyramid.
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


class Kaggle1SegImagePyramidViewConfBuilder(AbstractImagingViewConfBuilder):
    """Builder for Kaggle-1 segmentation mask datasets (vis-lifted with parent).

    Handles two cases:
    - Base images co-located in the entity's files (same as Kaggle-2 layout)
    - Base images only in the parent's support entity (must be fetched externally)

    The builder checks own files first. If base images are found in a known
    directory (lab_processed, processed_microscopy, etc.), it uses them directly.
    Otherwise, it looks up the parent's support entity for the base images.
    """

    def __init__(
        self, entity, groups_token, assets_endpoint, get_entity=None, parent=None, find_support_entity=None, **kwargs
    ):
        super().__init__(entity, groups_token, assets_endpoint, **kwargs)
        self._get_entity = get_entity
        self._parent_uuid = parent.get("uuid") if isinstance(parent, dict) else parent
        self._find_support_entity = find_support_entity
        self.seg_image_pyramid_regex = IMAGE_PYRAMID_DIR
        self.view_type = KAGGLE_IMAGE_VIEW_TYPE
        self._base_image_source = None

    @property
    def base_image_source(self):
        """How base images were resolved: 'colocated' or 'support_entity'.

        None if get_conf_cells() has not been called yet.
        """
        return self._base_image_source

    def _has_colocated_base_images(self):
        """Check if the entity has base images in its own files (Kaggle-2 style)."""
        file_paths_found = self._get_file_paths()
        paths = get_found_images_all(file_paths_found)
        matched_dirs = {dir for dir in base_image_dirs if any(dir in img for img in paths)}
        return matched_dirs

    def get_conf_cells(self, **kwargs):
        if self._parent_uuid is None:
            raise ValueError("Kaggle1SegImagePyramidViewConfBuilder requires a parent dataset")

        # Check if base images are co-located in own files
        matched_dirs = self._has_colocated_base_images()

        if matched_dirs:
            # Base images found locally — use Kaggle-2 style (co-located)
            self._base_image_source = "colocated"
            image_dir = next(iter(matched_dirs))
            self.image_pyramid_regex = f"{IMAGE_PYRAMID_DIR}/{image_dir}"
            return self.get_conf_cells_common(self._get_img_and_offset_url_seg, **kwargs)

        # No base images in own files — fetch from parent's support entity
        self._base_image_source = "support_entity"
        return self._get_conf_cells_from_support(**kwargs)

    def _get_conf_cells_from_support(self, **kwargs):
        """Generate config using base images from parent's support entity."""
        # 1. Find the parent's support entity (has base images)
        support_entity = self._resolve_support_entity()
        support_uuid = support_entity.get("uuid")

        # 2. Find base image in support entity's files
        support_files = support_entity.get("files", [])
        if not support_files and support_entity.get("metadata", {}).get("files"):
            support_files = support_entity["metadata"]["files"]
        support_file_paths = [f["rel_path"] for f in support_files]

        found_images = list(
            get_matches(
                support_file_paths,
                IMAGE_PYRAMID_DIR + r".*\.ome\.tiff?$",
            )
        )

        if not found_images:
            raise FileNotFoundError(f"Support entity {support_uuid} is missing base image pyramid files")

        # 3. Build URLs using support entity's UUID
        img_path = found_images[0]
        base_img_url = self._build_support_url(support_uuid, img_path)

        offsets_path = re.sub(
            r"ome\.tiff?",
            "offsets.json",
            re.sub(IMAGE_PYRAMID_DIR, OFFSETS_DIR, img_path),
        )
        base_offsets_url = self._build_support_url(support_uuid, offsets_path)

        metadata_path = re.sub(
            r"ome\.tiff?",
            "metadata.json",
            re.sub(IMAGE_PYRAMID_DIR, IMAGE_METADATA_DIR, img_path),
        )
        base_metadata_url = self._build_support_url(support_uuid, metadata_path)

        self.base_image_metadata = get_image_metadata(self, base_metadata_url)

        # 4. Create Vitessce config with base image from support entity
        vc, dataset = self._create_vitessce_config(dataset_name="Visualization Files")
        dataset = dataset.add_object(
            ImageOmeTiffWrapper(
                img_url=base_img_url,
                offsets_url=base_offsets_url,
                name=Path(img_path).stem,
            )
        )

        # 5. Add segmentation overlay from own entity files
        self._add_segmentation_image(dataset)

        # 6. Kaggle-style view setup
        conf = self._setup_view_config(vc, dataset, self.view_type).to_dict()
        return get_conf_cells(conf)

    def _resolve_support_entity(self):
        """Find the parent's support entity containing base images."""
        if self._find_support_entity is not None:
            support = self._find_support_entity(self._parent_uuid)
            if support is not None:
                return support

        raise ValueError(
            f"Kaggle1SegImagePyramidViewConfBuilder: could not find support entity for parent {self._parent_uuid}"
        )

    def _build_support_url(self, support_uuid, rel_path):
        """Build an assets URL for a file in the support entity."""
        import urllib.parse

        base_url = urllib.parse.urljoin(self._assets_endpoint, f"{support_uuid}/{rel_path}")
        if self._groups_token:
            token_param = urllib.parse.urlencode({"token": self._groups_token})
            return f"{base_url}?{token_param}"
        return base_url


class GeoMxImagePyramidViewConfBuilder(AbstractImagingViewConfBuilder):
    """Wrapper class for creating a view configuration for image pyramids for GeoMx datasets, that show,
    segmentation mask layered over a base image-pyramid with AOIs and ROIs highlighted.
    i.e for high resolution viz-lifted imaging datasets like
    https://portal.hubmapconsortium.org/browse/dataset/7a009a7cca74d63e2a9e184c6c1becca
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
        # ponytail: spatialBeta reads physical pixel size from OME metadata automatically,
        # so the old raster.json usePhysicalSizeScaling flag is no longer needed.


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
            raise FileNotFoundError(f"Dataset {self._uuid} is missing seqFish hybridization cycle image files")
        # Get all files grouped by PosN names.
        images_by_pos = group_by_file_name(found_images)
        confs = []
        # Build up a conf for each Pos.
        for images in images_by_pos:
            pos_name = self._get_pos_name(images[0])
            vc, dataset = self._create_vitessce_config(name=pos_name, dataset_name=pos_name)
            sorted_images = sorted(images, key=self._get_hybcycle)
            image_fileuids = [f"image_{i}" for i in range(len(sorted_images))]
            first_image_url = None
            for fileuid, img_path in zip(image_fileuids, sorted_images, strict=True):
                img_url, offsets_url, _ = self._get_img_and_offset_url(img_path, IMAGE_PYRAMID_DIR)
                first_image_url = first_image_url or img_url
                dataset.add_object(
                    ImageOmeTiffWrapper(
                        img_url=img_url,
                        offsets_url=offsets_url,
                        name=self._get_hybcycle(img_path),
                        coordination_values={"fileUid": fileuid},
                    )
                )
            # All hyb-cycle images of a position share an acquisition, so read channels once.
            image_channel_count = (get_ome_tiff_metadata(first_image_url) or {}).get("SizeC")
            vc = self._setup_view_config(
                vc,
                dataset,
                self.view_type,
                # Every hyb cycle covers the same field of view and all of them render at once,
                # so a volume load here would be one whole volume per cycle per channel.
                disable_3d=True,
                image_fileuids=image_fileuids,
                image_channel_count=image_channel_count,
            )
            confs.append(vc.to_dict())
        return get_conf_cells(confs)

    def _get_hybcycle(self, image_path):
        return re.search(SEQFISH_HYB_CYCLE_REGEX, image_path)[0]

    def _get_pos_name(self, image_path):
        return re.search(SEQFISH_FILE_REGEX, image_path)[0].split(".")[0]
