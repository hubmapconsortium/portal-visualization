from pathlib import Path
import re
from itertools import groupby

import nbformat
from vitessce import VitessceConfig
# from vitessce.config import VitessceConfigCoordinationScope

from .builders.base_builders import ConfCells


def get_matches(files, regex):
    return list(
        set(
            match[0] for match in set(re.search(regex, file) for file in files) if match
        )
    )


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


# TODO: This is commented out because we are currently using obssets to display marker gene labels
#
# def use_multiple_coordinations(vc, views, coordinationType, values):
#     """
#     A helper function for creating multiple coordination scopes for the same coordination type
#     Should no longer be necessary after https://github.com/vitessce/vitessce-python/issues/271
#     :param vc: The VitessceConfig object to add the coordinations to
#     :param views: The views to add the coordinations to
#     :param coordinationType: The coordination type for the scope
#     :param values: The values for the coordination scope
#     :return: The VitessceConfig object with the added coordinations
#     """
#     # Add the coordinations to the VC
#     coordinations = vc.add_coordination(*[coordinationType for _ in values])

#     # Set coordination values
#     for i, value in enumerate(values):
#         coordinations[i].set_value(value)

#     scopes = [str(l.c_scope) for l in coordinations]

#     custom_scope = VitessceConfigCoordinationScope(
#         c_type=coordinationType, c_scope=scopes)

#     for v in views:
#         v.use_coordination(custom_scope)

#     return vc
