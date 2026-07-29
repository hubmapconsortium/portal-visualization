import logging
import re
from itertools import groupby
from pathlib import Path
from unicodedata import normalize

import fsspec
import nbformat
import requests
import zarr
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.zip import ZipFileSystem
from vitessce import VitessceConfig

from .builders.base_builders import ConfCells
from .constants import PORTAL_VIS_USER_AGENT, image_units

logger = logging.getLogger(__name__)


def with_config_builder_user_agent(request_init):
    """Merge the config-builder User-Agent into a server-side request's headers.

    The back-end throttles scraping by User-Agent (default aiohttp/requests/urllib UAs); config
    building makes many server-side requests, so tag them with a distinctive UA the back-end can
    whitelist. Server-side only: the browser sets its own User-Agent and can't override it, so
    this is deliberately kept out of the browser-facing ``_get_request_init``.

    >>> with_config_builder_user_agent(None) == {'headers': {'User-Agent': PORTAL_VIS_USER_AGENT}}
    True
    >>> with_config_builder_user_agent({'headers': {'Authorization': 'Bearer x'}}) == {
    ...     'headers': {'User-Agent': PORTAL_VIS_USER_AGENT, 'Authorization': 'Bearer x'}}
    True
    """
    request_init = dict(request_init or {})
    request_init["headers"] = {"User-Agent": PORTAL_VIS_USER_AGENT, **request_init.get("headers", {})}
    return request_init


def get_matches(files, regex):
    return list({match[0] for match in {re.search(regex, file) for file in files} if match})


def create_coordination_values(obs_type="cell", **kwargs):
    return {"obsType": obs_type, **kwargs}


def _get_path_name(file):
    return Path(file).name


def group_by_file_name(files):
    sorted_files = sorted(files, key=_get_path_name)
    return [list(g) for _, g in groupby(sorted_files, _get_path_name)]


def get_conf_cells(vc_anything):
    cells = _get_cells_from_anything(vc_anything)
    conf = vc_anything.to_dict() if hasattr(vc_anything, "to_dict") else vc_anything
    return ConfCells(conf, cells)


def _get_cells_from_anything(vc):
    if isinstance(vc, dict):
        return _get_cells_from_dict(vc)
    if isinstance(vc, list):
        return _get_cells_from_list(vc)
    if hasattr(vc, "to_python"):
        return _get_cells_from_obj(vc)
    raise Exception(f"Viewconf is unexpected type {type(vc)}")  # pragma: no cover


def _get_cells_from_list(vc_list):
    cells = [nbformat.v4.new_markdown_cell("Multiple visualizations are available.")]
    for vc in vc_list:
        cells.extend(_get_cells_from_anything(vc))
    return cells


def _get_cells_from_dict(vc_dict):
    vc_obj = VitessceConfig.from_dict(vc_dict)
    return _get_cells_from_obj(vc_obj)


def _get_cells_from_obj(vc_obj):
    imports, conf_expression = vc_obj.to_python()
    return [
        nbformat.v4.new_code_cell(f"from vitessce import {', '.join(imports)}"),
        nbformat.v4.new_code_cell(f"conf = {conf_expression}\nconf.widget()"),
    ]


def get_found_images(image_pyramid_regex, file_paths_found):
    try:
        found_images = [
            path
            for path in get_matches(
                file_paths_found,
                image_pyramid_regex + r".*\.ome\.tiff?$",
            )
            if "separate/" not in path
        ]
        return found_images
    except Exception as e:
        raise RuntimeError(f"Error while searching for pyramid images: {e}")  # noqa: B904


def obs_has_column(zroot, col_name: str, obs_path: str = "obs") -> bool:
    """Return True if the raw column exists in obs_path"""
    try:
        grp = zroot[obs_path]
    except KeyError:  # pragma: no cover
        return False
    return col_name in grp


def get_found_images_all(file_paths_found):
    found_images = [
        path
        for path in get_matches(
            file_paths_found,
            r".*\.ome\.tiff?$",
        )
        if "separate/" not in path
    ]
    return found_images


def get_image_metadata(self, img_url):
    """
    Retrieve metadata from an image URL.
    >>> from unittest.mock import Mock, patch
    >>> mock_instance = Mock()
    >>> mock_instance._get_request_init.return_value = {}
    >>> mock_response = Mock()
    >>> mock_response.status_code = 404
    >>> mock_response.reason = 'Not Found'
    >>> with patch('requests.get', return_value=mock_response):
    ...     result = get_image_metadata(mock_instance, 'https://example.com/image')
    ...     assert result is None
    """

    meta_data = None
    request_init = with_config_builder_user_agent(self._get_request_init())
    response = requests.get(img_url, **request_init)
    if response.status_code == 200:  # pragma: no cover
        data = response.json()
        if isinstance(data, dict) and "PhysicalSizeX" in data and "PhysicalSizeUnitX" in data:
            meta_data = data
        else:
            logger.warning("Image does not have metadata")
    else:
        logger.warning("Failed to retrieve %s: %s - %s", img_url, response.status_code, response.reason)
    return meta_data


def get_ome_tiff_metadata(url):
    """Read pixel metadata from a (possibly remote) OME-TIFF's OME-XML.

    Reads only the TIFF header via range requests rather than downloading the image data. Returns a
    dict with pixel dimensions (``SizeX``/``SizeY``), channel count (``SizeC``), and physical pixel
    sizes in the shape ``get_image_scale`` expects (``PhysicalSizeX/Y`` plus ``PhysicalSizeUnitX/Y``),
    or None if it can't be read. The units matter: image and mask may use different units (e.g. µm vs
    mm), so the raw values must not be compared directly.

    >>> import tifffile, numpy as np, tempfile, os
    >>> path = os.path.join(tempfile.mkdtemp(), "t.ome.tif")
    >>> tifffile.imwrite(path, np.zeros((3, 30, 40), dtype=np.uint8), photometric="minisblack",
    ...                  metadata={"axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeXUnit": "µm",
    ...                            "PhysicalSizeY": 0.5, "PhysicalSizeYUnit": "µm"})
    >>> meta = get_ome_tiff_metadata(path)
    >>> (meta["SizeX"], meta["SizeY"], meta["SizeC"], meta["PhysicalSizeX"], meta["PhysicalSizeUnitX"])
    (40, 30, 3, 0.5, 'µm')
    >>> plain = os.path.join(tempfile.mkdtemp(), "plain.tif")
    >>> tifffile.imwrite(plain, np.zeros((2, 2), dtype=np.uint8), ome=False)
    >>> get_ome_tiff_metadata(plain) is None
    True
    """
    try:
        import xml.etree.ElementTree as ElementTree

        import tifffile

        # Send the config-builder User-Agent so these server-side range reads aren't throttled by the
        # assets back-end's scraping filter (the token, when present, rides in the URL query string).
        source = fsspec.open(url, client_kwargs=with_config_builder_user_agent(None))
        with source.open() as f, tifffile.TiffFile(f) as tif:
            ome_xml = tif.ome_metadata
        if not ome_xml:
            return None
        pixels = next((e for e in ElementTree.fromstring(ome_xml).iter() if e.tag.endswith("Pixels")), None)
        if pixels is None:  # pragma: no cover
            return None
        size_x, size_y, size_c = pixels.get("SizeX"), pixels.get("SizeY"), pixels.get("SizeC")
        physical_x, physical_y = pixels.get("PhysicalSizeX"), pixels.get("PhysicalSizeY")
        # Channel names in acquisition (index) order, used to map segmentation channel names to indices.
        channel_names = [c.get("Name") for c in pixels if c.tag.endswith("Channel")]
        return {
            "SizeX": int(size_x) if size_x else None,
            "SizeY": int(size_y) if size_y else None,
            "SizeC": int(size_c) if size_c else 1,
            "ChannelNames": channel_names,
            "PhysicalSizeX": float(physical_x) if physical_x else None,
            "PhysicalSizeY": float(physical_y) if physical_y else None,
            "PhysicalSizeUnitX": pixels.get("PhysicalSizeXUnit"),
            "PhysicalSizeUnitY": pixels.get("PhysicalSizeYUnit"),
        }
    except Exception as e:  # pragma: no cover
        logger.warning("Could not read OME-TIFF metadata from %s: %s", url, e)
        return None


# Physical-size unit spellings -> micrometers per unit.
_UNIT_TO_MICROMETERS = {
    "nm": 1e-3,
    "nanometer": 1e-3,
    "nanometre": 1e-3,
    "µm": 1.0,
    "μm": 1.0,
    "um": 1.0,
    "micron": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "mm": 1e3,
    "millimeter": 1e3,
    "millimetre": 1e3,
    "cm": 1e4,
    "dm": 1e5,
    "m": 1e6,
    "meter": 1e6,
    "metre": 1e6,
}


def _physical_size_micrometers(size, unit):
    """Physical pixel size in micrometers. Missing size -> 1.0 (Vitessce's per-pixel default);
    unknown unit -> the raw value (best effort)."""
    if not size:
        return 1.0
    if isinstance(unit, str):
        factor = _UNIT_TO_MICROMETERS.get(normalize("NFKC", unit).strip().lower())
        if factor:
            return float(size) * factor
    return float(size)


def get_segmentation_alignment_scale(image_metadata, mask_metadata):
    """5-element coordinate-transformation scale aligning a segmentation mask onto an image.

    Vitessce places layers by physical size, so the scale is the ratio of the image's physical
    pixel size to the mask's. Unit spellings are normalized and missing/unitless physical sizes are
    treated as 1 µm/px, so (unlike get_image_scale) it does not bail to identity when one OME-TIFF
    omits its unit. Returns identity only when metadata is entirely absent or degenerate.

    >>> img = {"PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5, "PhysicalSizeUnitX": "µm", "PhysicalSizeUnitY": "µm"}
    >>> mask = {"PhysicalSizeX": 0.000984, "PhysicalSizeY": 0.000984, "PhysicalSizeUnitX": "mm", \
                "PhysicalSizeUnitY": "mm"}
    >>> get_segmentation_alignment_scale(img, mask)
    [0.50813, 0.50813, 1, 1, 1]
    >>> get_segmentation_alignment_scale(img, {"PhysicalSizeX": None})  # mask lacks physical size -> 1 µm/px
    [0.5, 0.5, 1, 1, 1]
    >>> get_segmentation_alignment_scale(None, mask)
    [1, 1, 1, 1, 1]
    >>> get_segmentation_alignment_scale({"PhysicalSizeX": 2.0, "PhysicalSizeY": 2.0},
    ...                                  {"PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0})  # no units -> raw values
    [2.0, 2.0, 1, 1, 1]
    >>> get_segmentation_alignment_scale({"PhysicalSizeX": 508.0, "PhysicalSizeY": 508.0},
    ...                                  {"PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0})  # implausible -> identity
    [1, 1, 1, 1, 1]
    """
    if not image_metadata or not mask_metadata:
        return [1, 1, 1, 1, 1]
    mask_x = _physical_size_micrometers(mask_metadata.get("PhysicalSizeX"), mask_metadata.get("PhysicalSizeUnitX"))
    mask_y = _physical_size_micrometers(mask_metadata.get("PhysicalSizeY"), mask_metadata.get("PhysicalSizeUnitY"))
    if not (mask_x and mask_y):  # pragma: no cover
        return [1, 1, 1, 1, 1]
    image_x = _physical_size_micrometers(image_metadata.get("PhysicalSizeX"), image_metadata.get("PhysicalSizeUnitX"))
    image_y = _physical_size_micrometers(image_metadata.get("PhysicalSizeY"), image_metadata.get("PhysicalSizeUnitY"))
    scale_x, scale_y = round(image_x / mask_x, 5), round(image_y / mask_y, 5)
    # Guard against an unresolved unit mismatch producing an absurd scale that would exhaust GPU memory.
    if not all(0.01 <= s <= 100 for s in (scale_x, scale_y)):
        logger.warning("Discarding implausible segmentation scale %s; rendering unscaled.", [scale_x, scale_y])
        return [1, 1, 1, 1, 1]
    return [scale_x, scale_y, 1, 1, 1]


def get_image_scale(base_metadata, seg_metadata):
    """
    Computes the scale between two image metadata based on physical size.

    Args:
        base_metadata (dict): Metadata for the base image.
        seg_metadata (dict): Metadata for the segmented image.

    Returns:
        list: A list containing the scale factors for x, y, while keeping others unchanged (as 1).

    Doctest:
    >>> base_metadata = { \
        'PhysicalSizeX': 50, 'PhysicalSizeY': 100, 'PhysicalSizeUnitX': 'mm', 'PhysicalSizeUnitY': 'mm' \
    }
    >>> seg_metadata = { \
        'PhysicalSizeX': 25, 'PhysicalSizeY': 50, 'PhysicalSizeUnitX': 'mm', 'PhysicalSizeUnitY': 'mm' \
    }
    >>> scale = get_image_scale(base_metadata, seg_metadata)
    >>> assert scale == [2.0, 2.0, 1, 1, 1]

    >>> base_metadata = { \
        'PhysicalSizeX': 50, 'PhysicalSizeY': 100, 'PhysicalSizeUnitX': 'mm', 'PhysicalSizeUnitY': 'mm' \
    }
    >>> seg_metadata = None
    >>> scale = get_image_scale(base_metadata, seg_metadata)
    >>> assert scale == [1, 1, 1, 1, 1]
    """

    scale = [1, 1, 1, 1, 1]
    seg_x, seg_y, seg_x_unit, seg_y_unit = None, None, None, None
    base_x, base_y, base_x_unit, base_y_unit = None, None, None, None

    if seg_metadata is not None:
        seg_x, seg_y, seg_x_unit, seg_y_unit = get_physical_size_units(seg_metadata)

    if base_metadata is not None:
        base_x, base_y, base_x_unit, base_y_unit = get_physical_size_units(base_metadata)

    if all([base_x_unit, base_y_unit, seg_x_unit, seg_y_unit]) and all(
        unit in image_units for unit in [base_x_unit, base_y_unit, seg_x_unit, seg_y_unit]
    ):
        scale_x = (float(base_x) / float(seg_x)) * (image_units[seg_x_unit] / image_units[base_x_unit])  # type: ignore
        scale_y = (float(base_y) / float(seg_y)) * (image_units[seg_y_unit] / image_units[base_y_unit])  # type: ignore

        scale = [round(scale_x, 5), round(scale_y, 5), 1, 1, 1]
    else:
        logger.warning("PhysicalSize units are not correct")
    logger.debug("Scaling factor: %s", scale)
    return scale


def get_physical_size_units(metadata):
    """
        Extracts the physical size units (X, Y) from metadata.

        Args:
            metadata (dict): The metadata dictionary for the image.

        Returns:
            tuple: A tuple containing the physical sizes and their respective units.

        Doctest:

        >>> metadata = { \
            'PhysicalSizeX': 50, 'PhysicalSizeY': 100, 'PhysicalSizeUnitX': 'mm', 'PhysicalSizeUnitY': 'mm' \
        }
        >>> get_physical_size_units(metadata)
        (50, 100, 'mm', 'mm')

        >>> metadata = { \
            'PhysicalSizeX': None, 'PhysicalSizeY': 100, 'PhysicalSizeUnitX': 'mm', 'PhysicalSizeUnitY': 'mm' \
        }
        >>> get_physical_size_units(metadata)
        (1, 100, 'mm', 'mm')
    """

    # size_x and size_y will be one if nothing is provided or explicitly None
    size_x = metadata.get("PhysicalSizeX") or 1
    size_y = metadata.get("PhysicalSizeY") or 1
    size_x_unit = convert_unicode_unit(metadata, "PhysicalSizeUnitX")
    size_y_unit = convert_unicode_unit(metadata, "PhysicalSizeUnitY")

    return size_x, size_y, size_x_unit, size_y_unit


def convert_unicode_unit(metadata, key):
    """
    Converts any unicode string (e.g., representing image units) in the metadata key to a normalized format.

    Args:
        metadata (dict): The metadata dictionary containing the key.
        key (str): The key for the unit (e.g., 'PhysicalSizeUnitX').

    Returns:
        str or None: The normalized unit as a string, or None if not found.

    Doctest:

    >>> metadata = {'PhysicalSizeUnitX': 'mm'}
    >>> convert_unicode_unit(metadata, 'PhysicalSizeUnitX')
    'mm'

    >>> metadata = {'PhysicalSizeUnitY': '\u00b5m'}
    >>> convert_unicode_unit(metadata, 'PhysicalSizeUnitY')
    'μm'

    >>> metadata = {'PhysicalSizeUnitY': None}
    >>> convert_unicode_unit(metadata, 'PhysicalSizeUnitY')
    """
    # Check if the key exists and if the value is a string
    if key in metadata and isinstance(metadata[key], str):
        # Normalize the unicode string
        return normalize("NFKC", metadata[key])

    # Return None if the key is not present or the value isn't a string
    return None


def files_from_response(response_json):
    """
    >>> response_json = {'hits': {'hits': [
    ...     {
    ...         '_id': '1234',
    ...         '_source': {
    ...             'files': [{
    ...                 'rel_path': 'abc.txt'
    ...             }]
    ...         }
    ...     }
    ... ]}}
    >>> files_from_response(response_json)
    {'1234': ['abc.txt']}
    """
    hits = response_json["hits"]["hits"]
    return {hit["_id"]: [file["rel_path"] for file in hit["_source"].get("files", [])] for hit in hits}


class _SafeZipFileSystem(ZipFileSystem):
    """ZipFileSystem that raises FileNotFoundError (not KeyError) for absent members.

    zarr v3 stores require ``get`` to return ``None`` for a missing key; it only
    translates ``FileNotFoundError`` to ``None``. ZipFileSystem raises ``KeyError``,
    which otherwise breaks zarr's probes for optional metadata (e.g. ``.zmetadata``).
    """

    def _open(self, path, mode="rb", *args, **kwargs):
        try:
            return super()._open(path, mode, *args, **kwargs)
        except KeyError as e:
            raise FileNotFoundError(path) from e


def read_zip_zarr(zarr_url, request_init):
    """
    Opens a (possibly remote) zip-format zarr store via fsspec, range-reading rather
    than downloading the whole archive.

    Parameters:
        zarr_url (str): URL to the zipped .zarr file.
        request_init (dict): Client kwargs for request customization.

    Returns:
        zarr.Group: Opened Zarr store.
    """
    # ZipFileSystem names these target_protocol/target_options (remote_* is ReferenceFileSystem's
    # spelling and gets silently swallowed by **kwargs, dropping the auth header -> 403 on any
    # non-public asset).
    fs = _SafeZipFileSystem(
        fo=zarr_url,
        target_protocol="https",
        target_options={"client_kwargs": with_config_builder_user_agent(request_init)},
    )
    # zarr v3 needs an async-capable store; wrap the sync zip filesystem.
    store = zarr.storage.FsspecStore(AsyncFileSystemWrapper(fs, asynchronous=True), read_only=True)
    return zarr.open_group(store, mode="r", use_consolidated=False)


def _listing_forbidden_or_missing(exc):
    """True for the responses the assets server gives when a directory can't be listed."""
    # aiohttp raises ClientResponseError with a numeric .status; fsspec may wrap as FileNotFoundError.
    return getattr(exc, "status", None) in (403, 404) or isinstance(exc, (FileNotFoundError, PermissionError))


class _SafeHTTPFileSystem(HTTPFileSystem):
    """HTTPFileSystem that treats a forbidden/absent directory listing as empty.

    The HuBMAP assets server serves individual files but returns 403 for directory GETs (no
    listing). zarr v3 enumerates group members by listing the directory (zarr v2 read keys by
    path and never listed), so a plain open crashes on published stores with a 403 on e.g.
    ``.../obs/``. Returning an empty listing lets member enumeration degrade to "nothing found"
    while direct key reads -- how anndata arrays/attrs are actually accessed -- keep working.
    """

    async def _ls(self, url, detail=True, **kwargs):
        # zarr's list_dir() -> fs._ls(); used for group member enumeration.
        try:
            return await super()._ls(url, detail=detail, **kwargs)
        except Exception as exc:
            if _listing_forbidden_or_missing(exc):
                return []
            raise

    async def _find(self, path, *args, **kwargs):
        # zarr's list() / list_prefix() -> fs._find().
        try:
            return await super()._find(path, *args, **kwargs)
        except Exception as exc:
            if _listing_forbidden_or_missing(exc):
                return []
            raise


def read_zarr(zarr_url, request_init):
    """Open a (non-zip) zarr store over HTTP, tolerating assets servers that forbid directory
    listing. The builders only ever pass an assets (https) URL here.

    Requires zarr v3 (like ``read_zip_zarr``); the runtime environments are expected to install
    the pinned vitessce/zarr v3 dependencies.

    Parameters:
        zarr_url (str): URL to the .zarr store.
        request_init (dict): Client kwargs for request customization.

    Returns:
        zarr.Group: Opened Zarr store.
    """
    fs = _SafeHTTPFileSystem(asynchronous=True, client_kwargs=with_config_builder_user_agent(request_init))
    store = zarr.storage.FsspecStore(fs, path=zarr_url, read_only=True)
    return zarr.open_group(store, mode="r", use_consolidated=False)
