#!/usr/bin/env python3
import argparse
import yaml
import json
from pathlib import Path
from os import environ
from dataclasses import dataclass

import pytest
import zarr

from src.portal_visualization.epic_factory import get_epic_builder
from src.portal_visualization.builders.base_builders import ConfCells
from src.portal_visualization.builder_factory import (
    get_view_config_builder,
    has_visualization,
)


def str_presenter(dumper, data):
    # From https://stackoverflow.com/a/33300001
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)


@dataclass
class MockResponse:
    content: str


good_entity_paths = list(
    (Path(__file__).parent / "good-fixtures").glob("*/*-entity.json")
)
assert len(good_entity_paths) > 0

image_pyramids = [
    "IMSViewConfBuilder",
    "SeqFISHViewConfBuilder",
    "NanoDESIViewConfBuilder",
]

image_pyramid_paths = [
    path for path in good_entity_paths if path.parent.name in image_pyramids
]
assert len(image_pyramid_paths) > 0

bad_entity_paths = list((Path(__file__).parent / "bad-fixtures").glob("*-entity.json"))
assert len(bad_entity_paths) > 0

assaytypes_path = Path(__file__).parent / "assaytype-fixtures"
assert assaytypes_path.is_dir()

defaults = json.load((Path(__file__).parent.parent / "src/defaults.json").open())

default_assaytype = {
    "soft_assaytype": "Null",
    "vitessce-hints": [],
}


def get_entity(input):
    # uuid = entity.get("uuid")
    if not isinstance(input, str):
        uuid = input.get("uuid")
    else:
        uuid = input
    # print(uuid, assaytypes_path.joinpath(f"{uuid}.json"))
    if uuid is None:  # pragma: no cover
        return default_assaytype
    assay = json.loads(assaytypes_path.joinpath(f"{uuid}.json").read_text())
    return assay


@pytest.mark.parametrize(
    "has_vis_entity",
    [
        (False, {"uuid": "2c2179ea741d3bbb47772172a316a2bf"}),
        (True, json.loads(Path.read_text(good_entity_paths[0]))),
        # NOTE: If the first fixture returns a Null builder this breaks.
    ],
    ids=lambda has_vis_entity: f"has_visualization={has_vis_entity[0]}",
)
def test_has_visualization(has_vis_entity):
    has_vis, entity = has_vis_entity
    parent = entity.get("parent") or None  # Only used for image pyramids
    # TODO: Once other epic hints exist, this may need to be adjusted
    epic_uuid = (
        entity.get("uuid")
        if "segmentation_mask" in entity.get("vitessce-hints", {})
        else None
    )
    assert has_vis == has_visualization(entity, get_entity, parent, epic_uuid)


def mock_zarr_store(entity_path, mocker):
    # Need to mock zarr.open to yield correct values for different scenarios
    z = zarr.open()
    gene_array = zarr.array(["ENSG00000139618", "ENSG00000139619", "ENSG00000139620"])
    is_annotated = "is-annotated" in entity_path.name
    is_multiome = "multiome" in entity_path.name
    if is_multiome:
        obs = z.create_group("mod/rna/obs")
        var = z.create_group("mod/rna/var")
        group_names = ["leiden_wnn", "leiden_rna", "cluster_cbg", "cluster_cbb"]
        if is_annotated:
            group_names.append("predicted_label")
        groups = obs.create_groups(*group_names)
        for group in groups:
            group["categories"] = zarr.array(["0", "1", "2"])

    obs = z.create_group("obs")
    obs["marker_gene_0"] = gene_array
    if is_annotated:
        path = (
            f'{"mod/rna/" if is_multiome else ""}uns/annotation_metadata/is_annotated'
        )
        z[path] = True
        if "asct" in entity_path.name:
            z["obs/predicted.ASCT.celltype"] = (
                True  # only checked for membership in zarr group
            )
        elif "predicted-label" in entity_path.name:
            z["obs/predicted_label"] = True  # only checked for membership in zarr group
            z["obs/predicted_CLID"] = True
    if "marker" in entity_path.name:
        obs.attrs["encoding-version"] = "0.1.0"
        var = z.create_group("var")
        var.attrs["_index"] = "index"
        var["index"] = gene_array
        var["hugo_symbol"] = zarr.array([0, 1, 2])
        var["hugo_symbol"].attrs["categories"] = "hugo_categories"
        var["hugo_categories"] = zarr.array(["gene123", "gene456", "gene789"])
    if "visium" in entity_path.name:
        z["uns/spatial/visium/scalefactors/spot_diameter_micrometers"] = 200.0
    mocker.patch("zarr.open", return_value=z)


@pytest.mark.parametrize(
    "entity_path", good_entity_paths, ids=lambda path: f"{path.parent.name}/{path.name}"
)
def test_entity_to_vitessce_conf(entity_path, mocker):
    mock_zarr_store(entity_path, mocker)

    possible_marker = entity_path.name.split("-")[-2]
    marker = (
        possible_marker.split("=")[1] if possible_marker.startswith("marker=") else None
    )
    epic_uuid = None
    entity = json.loads(entity_path.read_text())
    parent = entity.get("parent") or None  # Only used for image pyramids
    assay_type = get_entity(entity["uuid"])
    if "segmentation_mask" in assay_type["vitessce-hints"]:
        epic_uuid = entity.get("uuid")
    Builder = get_view_config_builder(entity, get_entity, parent, epic_uuid)
    # Envvars should not be set during normal test runs,
    # but to test the end-to-end integration, they are useful.
    groups_token = environ.get("GROUPS_TOKEN", "groups_token")
    assets_url = environ.get("ASSETS_URL", "https://example.com")
    # epic_uuid = environ.get("EPIC_UUID", "epic_uuid")
    builder = Builder(entity, groups_token, assets_url)
    conf, cells = builder.get_conf_cells(marker=marker)
    if "segmentation_mask" not in assay_type["vitessce-hints"]:
        assert Builder.__name__ == entity_path.parent.name
        compare_confs(entity_path, conf, cells)
    if "segmentation_mask" in assay_type["vitessce-hints"]:
        epic_builder = get_epic_builder(epic_uuid)
        assert epic_builder is not None
        assert epic_builder.__name__ == entity_path.parent.name
        if conf is None:  # pragma: no cover
            with pytest.raises(ValueError):
                epic_builder(
                    epic_uuid, ConfCells(conf, cells), entity, groups_token, assets_url
                ).get_conf_cells()
            return

        built_epic_conf, cells = epic_builder(
            epic_uuid, ConfCells(conf, cells), entity, groups_token, assets_url
        ).get_conf_cells()
        assert built_epic_conf is not None

        compare_confs(entity_path, built_epic_conf, cells)


@pytest.mark.parametrize("entity_path", bad_entity_paths, ids=lambda path: path.name)
def test_entity_to_error(entity_path, mocker):
    mock_zarr_store(entity_path, mocker)

    entity = json.loads(entity_path.read_text())
    with pytest.raises(Exception) as error_info:
        parent = entity.get("parent") or None  # Only used for image pyramids
        Builder = get_view_config_builder(entity, get_entity, parent=parent)
        builder = Builder(entity, "groups_token", "https://example.com/")
        builder.get_conf_cells()
    actual_error = f"{error_info.type.__name__}: {error_info.value.args[0]}"

    error_expected_path = entity_path.parent / entity_path.name.replace(
        "-entity.json", "-error.txt"
    )
    expected_error = error_expected_path.read_text().strip()
    assert actual_error == expected_error


def clean_cells(cells):
    return [
        {
            k: v
            for k, v in dict(c).items()
            if k not in {"metadata", "id", "execution_count", "outputs"}
        }
        for c in cells
    ]


def compare_confs(entity_path, conf, cells):
    expected_conf_path = entity_path.parent / entity_path.name.replace(
        "-entity", "-conf"
    )
    expected_conf = json.loads(expected_conf_path.read_text())

    # Compare normalized JSON strings so the diff is easier to read,
    # and there are fewer false positives.
    assert json.dumps(conf, indent=2, sort_keys=True) == json.dumps(
        expected_conf, indent=2, sort_keys=True
    )

    expected_cells_path = entity_path.parent / entity_path.name.replace(
        "-entity.json", "-cells.yaml"
    )
    if expected_cells_path.is_file():
        expected_cells = yaml.safe_load(expected_cells_path.read_text())

        # Compare as YAML to match fixture.
        assert yaml.dump(clean_cells(cells)) == yaml.dump(expected_cells)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate fixtures")
    parser.add_argument("--input", required=True, type=Path, help="Input JSON path")

    args = parser.parse_args()
    entity = json.loads(args.input.read_text())
    Builder = get_view_config_builder(entity, get_entity)
    builder = Builder(entity, "groups_token", "https://example.com/")
    conf, cells = builder.get_conf_cells()

    print(yaml.dump(clean_cells(cells), default_style="|"))
