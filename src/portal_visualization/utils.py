from pathlib import Path
import re
from itertools import groupby

import nbformat


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


def _get_cells_from_conf_list(confs):
    cells = []
    if len(confs) > 1:
        cells.append(nbformat.v4.new_markdown_cell('Multiple visualizations are available.'))
    for conf in confs:
        cells.extend(_get_cells_from_conf(conf))
    return cells


def _get_cells_from_conf(conf):
    imports, conf_expression = conf.to_python()
    return [
        nbformat.v4.new_code_cell(f'from vitessce import {", ".join(imports)}'),
        nbformat.v4.new_code_cell(f'conf = {conf_expression}\nconf.widget()'),
    ]
