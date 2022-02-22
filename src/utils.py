from pathlib import Path
import re
from itertools import groupby

import nbformat

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


def get_conf_cells(vc, md):
    imports, vc_code = vc.to_python()
    cells = [
        nbformat.v4.new_markdown_cell(md),
        nbformat.v4.new_code_cell(f'from vitessce import ${", ".join(imports)}'),
        nbformat.v4.new_code_cell(vc_code)
    ]
    # notebook = nbformat.v4.new_notebook(cells=cells)
    # nbformat.writes(notebook)
    return ConfCells(vc.to_dict(), cells)
