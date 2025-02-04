from pathlib import Path
import re
from itertools import groupby
from requests import get

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


def get_image_metadata(self, img_url):
    meta_data = None
    request_init = self._get_request_init() or {}
    response = get(img_url, **request_init)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, dict) and "PhysicalSizeX" in data and data['PhysicalSizeX'] is not None:
            meta_data = data
        else:
            print("Image does not have Physical sizes ")
    else:
        print(f"Failed to retrieve {img_url}: {response.status_code} - {response.reason}")
    return meta_data


def get_image_scale(base_metadata, seg_metadata):
    scale = [1, 1, 1, 1, 1]
    if seg_metadata is None and base_metadata is not None:
        if base_metadata['PhysicalSizeUnitX'] in image_units and base_metadata['PhysicalSizeUnitY'] in image_units:
            scale_x = base_metadata['PhysicalSizeX'] / image_units[base_metadata['PhysicalSizeUnitX']]
            scale_y = base_metadata['PhysicalSizeY'] / image_units[base_metadata['PhysicalSizeUnitY']]

            scale = [scale_x, scale_y, 1, 1, 1]
    return scale


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
