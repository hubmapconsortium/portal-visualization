from pathlib import Path
import re
from itertools import groupby
from requests import get
from unicodedata import normalize
import numpy as np

import nbformat
from vitessce import VitessceConfig

from .builders.base_builders import ConfCells
from .constants import image_units


def get_matches(files, regex):
    return list(
        set(
            match[0] for match in set(re.search(regex, file) for file in files) if match
        )
    )


def create_coordination_values(obs_type='cell', **kwargs):
    return {
        'obsType': obs_type,
        **kwargs
    }


def _get_path_name(file):
    return Path(file).name


def group_by_file_name(files):
    sorted_files = sorted(files, key=_get_path_name)
    return [list(g) for _, g in groupby(sorted_files, _get_path_name)]


def get_conf_cells(vc_anything):
    cells = _get_cells_from_anything(vc_anything)
    conf = (
        vc_anything.to_dict()
        if hasattr(vc_anything, 'to_dict')
        else vc_anything
    )
    return ConfCells(conf, cells)


def _get_cells_from_anything(vc):
    if isinstance(vc, dict):
        return _get_cells_from_dict(vc)
    if isinstance(vc, list):
        return _get_cells_from_list(vc)
    if hasattr(vc, 'to_python'):
        return _get_cells_from_obj(vc)
    raise Exception(f'Viewconf is unexpected type {type(vc)}')  # pragma: no cover


def _get_cells_from_list(vc_list):
    cells = [nbformat.v4.new_markdown_cell('Multiple visualizations are available.')]
    for vc in vc_list:
        cells.extend(_get_cells_from_anything(vc))
    return cells


def _get_cells_from_dict(vc_dict):
    vc_obj = VitessceConfig.from_dict(vc_dict)
    return _get_cells_from_obj(vc_obj)


def _get_cells_from_obj(vc_obj):
    imports, conf_expression = vc_obj.to_python()
    return [
        nbformat.v4.new_code_cell(f'from vitessce import {", ".join(imports)}'),
        nbformat.v4.new_code_cell(f'conf = {conf_expression}\nconf.widget()'),
    ]


def get_found_images(image_pyramid_regex, file_paths_found):
    found_images = [
        path for path in get_matches(
            file_paths_found, image_pyramid_regex + r".*\.ome\.tiff?$",
        )
        if 'separate/' not in path
    ]
    return found_images


def get_found_images_all(file_paths_found):
    found_images = [
        path for path in get_matches(
            file_paths_found, r".*\.ome\.tiff?$",
        )
        if 'separate/' not in path
    ]
    return found_images


def get_image_metadata(self, img_url):
    """
    Retrieve metadata from an image URL.
    >>> import builtins
    >>> from unittest.mock import Mock, patch
    >>> mock_instance = Mock()
    >>> mock_instance._get_request_init.return_value = {}
    >>> mock_response = Mock()
    >>> mock_response.status_code = 404
    >>> mock_response.reason = 'Not Found'
    >>> with patch('requests.get', return_value=mock_response):
    ...     with patch.object(builtins, 'print') as mock_print:
    ...         result = get_image_metadata(mock_instance, 'https://example.com/image')
    ...         mock_print.assert_called_with(f"Failed to retrieve https://example.com/image: 404 - Not Found")
    ...         assert result is None
    """

    meta_data = None
    request_init = self._get_request_init() or {}
    response = get(img_url, **request_init)
    if response.status_code == 200:  # pragma no cover
        data = response.json()
        if isinstance(data, dict) and "PhysicalSizeX" in data and 'PhysicalSizeUnitX' in data:
            meta_data = data
        else:
            print("Image does not have metadata")
    else:
        print(f"Failed to retrieve {img_url}: {response.status_code} - {response.reason}")
    return meta_data


def get_image_scale(base_metadata, seg_metadata):
    """
    Computes the scale between two image metadata based on physical size.

    Args:
        base_metadata (dict): Metadata for the base image.
        seg_metadata (dict): Metadata for the segmented image.

    Returns:
        list: A list containing the scale factors for x, y, while keeping others unchanged (as 1).

    Doctest:
    >>> from unittest.mock import Mock, patch
    >>> import builtins
    >>> base_metadata = { \
        'PhysicalSizeX': 50, 'PhysicalSizeY': 100, 'PhysicalSizeUnitX': 'mm', 'PhysicalSizeUnitY': 'mm' \
    }
    >>> seg_metadata = { \
        'PhysicalSizeX': 25, 'PhysicalSizeY': 50, 'PhysicalSizeUnitX': 'mm', 'PhysicalSizeUnitY': 'mm' \
    }
    >>> with patch('builtins.print') as mock_print:
    ...     scale = get_image_scale(base_metadata, seg_metadata)
    ...     mock_print.assert_called_with("Scaling factor: ", [2.0, 2.0, 1, 1, 1])
    ...     assert scale == [2.0, 2.0, 1, 1, 1]  # Ensure the return value is also correct

    >>> base_metadata = { \
        'PhysicalSizeX': 50, 'PhysicalSizeY': 100, 'PhysicalSizeUnitX': 'mm', 'PhysicalSizeUnitY': 'mm' \
    }
    >>> seg_metadata = None
    >>> with patch('builtins.print') as mock_print:
    ...     scale = get_image_scale(base_metadata, seg_metadata)
    ...     mock_print.assert_called_with("Scaling factor: ", [1, 1, 1, 1, 1])
    ...     assert scale == [1, 1, 1, 1, 1]  # Ensure the return value is also correct
    """

    scale = [1, 1, 1, 1, 1]
    seg_x, seg_y, seg_x_unit, seg_y_unit = None, None, None, None
    base_x, base_y, base_x_unit, base_y_unit = None, None, None, None

    if seg_metadata is not None:
        seg_x, seg_y, seg_x_unit, seg_y_unit = get_physical_size_units(seg_metadata)

    if base_metadata is not None:
        base_x, base_y, base_x_unit, base_y_unit = get_physical_size_units(base_metadata)

    if all([base_x_unit, base_y_unit, seg_x_unit, seg_y_unit]) and \
            all([unit in image_units for unit in [base_x_unit, base_y_unit, seg_x_unit, seg_y_unit]]):
        scale_x = (float(base_x) / float(seg_x)) * (image_units[seg_x_unit] / image_units[base_x_unit])
        scale_y = (float(base_y) / float(seg_y)) * (image_units[seg_y_unit] / image_units[base_y_unit])

        scale = [round(scale_x, 5), round(scale_y, 5), 1, 1, 1]
    else:
        print("PhysicalSize units are not correct")
    print("Scaling factor: ", scale)
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

    # size_x and size_y will be one if nothing is provided
    size_x = metadata['PhysicalSizeX'] if metadata['PhysicalSizeX'] is not None else 1
    size_y = metadata['PhysicalSizeY'] if metadata['PhysicalSizeY'] is not None else 1
    size_x_unit = convert_unicode_unit(metadata, 'PhysicalSizeUnitX')
    size_y_unit = convert_unicode_unit(metadata, 'PhysicalSizeUnitY')

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
        return normalize('NFKC', metadata[key])

    # Return None if the key is not present or the value isn't a string
    return None


def files_from_response(response_json):
    '''
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
    '''
    hits = response_json['hits']['hits']
    return {
        hit['_id']: [
            file['rel_path'] for file in hit['_source'].get('files', [])
        ] for hit in hits
    }


def get_spots_scaling_factor(z_store):
    """
    Computes the scaling factor for spot sizes in Visium spatial data.

    >>> store = {
    ...     'uns': {
    ...         'spatial/visium/scalefactors/spot_diameter_micrometers': np.array(65, dtype=np.int64),
    ...         'spatial/visium/scalefactors/spot_diameter_fullres': np.array(143.1718061674009, dtype=np.float64)
    ...     }
    ... }
    >>> get_spots_scaling_factor(store)
    2.20264

    >>> store_missing_key = {'uns': {
    ...     'spatial/visium/scalefactors/spot_diameter_fullres': np.array(143.1718061674009, dtype=np.float64)
    ... }}
    >>> get_spots_scaling_factor(store_missing_key)
    Warning: Missing keys in zarr store. Expected spot_diameter_micrometers & spot_diameter_fullres.
    1
    >>> store_zero_division = {'uns': {
    ...     'spatial/visium/scalefactors/spot_diameter_micrometers': np.array(0, dtype=np.int64),
    ...     'spatial/visium/scalefactors/spot_diameter_fullres': np.array(143.1718061674009, dtype=np.float64)
    ... }}
    >>> get_spots_scaling_factor(store_zero_division)
    Warning: Invalid or zero values in scalefactors. Using default scale factor.
    1

    >>> store_missing_value = {'uns': {
    ...    'spatial/visium/scalefactors/spot_diameter_micrometers': "invalid_type",
    ...    'spatial/visium/scalefactors/spot_diameter_fullres': 143.1718061674009
    ... }}
    >>> get_spots_scaling_factor(store_missing_value)
    Error occurred while computing the scale factor for spots: string indices must be integers
    1

    """
    scale_factor = 1
    diameter = 'spatial/visium/scalefactors/spot_diameter_micrometers'
    res = 'spatial/visium/scalefactors/spot_diameter_fullres'
    try:
        if diameter in z_store.get('uns', {}) and res in z_store.get('uns', {}):
            diameter_val = z_store['uns'][diameter][()]
            res_val = z_store['uns'][res][()]
            diameter_val = float(diameter_val) if isinstance(
                diameter_val, (np.integer, np.floating)) else diameter_val
            res_val = float(res_val) if isinstance(res_val, (np.integer, np.floating)) else res_val
            if isinstance(diameter_val, (int, float)) and isinstance(res_val, (int, float)) and diameter_val != 0:
                scale_factor = round(res_val / diameter_val, 5)
            else:
                print("Warning: Invalid or zero values in scalefactors. Using default scale factor.")

        else:
            print("Warning: Missing keys in zarr store. Expected spot_diameter_micrometers & spot_diameter_fullres.")

    except Exception as e:
        print("Error occurred while computing the scale factor for spots:", e)

    return scale_factor
