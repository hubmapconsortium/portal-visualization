#!/usr/bin/env python3
import argparse
import yaml
import json
from pathlib import Path
from os import environ
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import zarr

from src.portal_visualization.utils import get_found_images
from src.portal_visualization.epic_factory import get_epic_builder
from src.portal_visualization.builders.base_builders import ConfCells
from src.portal_visualization.builders.imaging_builders import KaggleSegImagePyramidViewConfBuilder
from src.portal_visualization.builder_factory import (
    get_view_config_builder,
    has_visualization,
)
from src.portal_visualization.paths import IMAGE_PYRAMID_DIR

groups_token = environ.get("GROUPS_TOKEN", "groups_token")
assets_url = environ.get("ASSETS_URL", "https://example.com")


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


test_cases = [
    (False, {"uuid": "2c2179ea741d3bbb47772172a316a2bf"}),
    (False, {"uuid": "f9ae931b8b49252f150d7f8bf1d2d13f-bad"}),
]

excluded_uuids = {entity["uuid"] for _, entity in test_cases}

for path in good_entity_paths:
    entity = json.loads(path.read_text())
    uuid = entity.get("uuid")
    if uuid in excluded_uuids:
        continue
    test_cases.append((True, entity))


@pytest.mark.parametrize(
    "has_vis_entity",
    test_cases,
    ids=lambda e: (
        f"has_visualization={e[0]}_uuid={e[1].get('uuid', 'no-uuid')}"
        if isinstance(e, tuple) else str(e)
    )
)
def test_has_visualization(has_vis_entity):
    has_vis, entity = has_vis_entity
    parent = entity.get("parent") or None  # Only used for image pyramids
    # TODO: Once other epic hints exist, this may need to be adjusted
    epic_uuid = (
        entity.get("uuid")
        if "epic" in entity.get("vitessce-hints", {})
        else None
    )
    result = has_visualization(entity, get_entity, parent, epic_uuid)
    print(f"UUID={entity.get('uuid')} expected={has_vis}, actual={result}")
    assert has_vis == result


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
    if "epic" in assay_type["vitessce-hints"]:
        epic_uuid = entity.get("uuid")
    Builder = get_view_config_builder(entity, get_entity, parent, epic_uuid)
    # Envvars should not be set during normal test runs,
    # but to test the end-to-end integration, they are useful.
    # epic_uuid = environ.get("EPIC_UUID", "epic_uuid")
    builder = Builder(entity, groups_token, assets_url)
    conf, cells = builder.get_conf_cells(marker=marker)
    if "epic" not in assay_type["vitessce-hints"]:
        assert Builder.__name__ == entity_path.parent.name
        compare_confs(entity_path, conf, cells)
    if "epic" in assay_type["vitessce-hints"]:
        epic_builder = get_epic_builder(epic_uuid)
        assert epic_builder is not None
        assert epic_builder.__name__ == entity_path.parent.name
        if conf is None:  # pragma: no cover
            with pytest.raises(ValueError):
                epic_builder(
                    epic_uuid, ConfCells(conf, cells), entity, groups_token, assets_url, builder.base_image_metadata
                ).get_conf_cells()
            return

        built_epic_conf, cells = epic_builder(
            epic_uuid, ConfCells(conf, cells), entity, groups_token, assets_url, builder.base_image_metadata
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


@pytest.fixture
def mock_seg_image_pyramid_builder():
    class MockBuilder(KaggleSegImagePyramidViewConfBuilder):
        def _get_file_paths(self):
            return []

    entity = json.loads(
        next(
            (Path(__file__).parent / "good-fixtures")
            .glob("KaggleSegImagePyramidViewConfBuilder/*-entity.json")
        ).read_text()
    )
    return MockBuilder(entity, groups_token, assets_url)


def test_filtered_images_not_found(mock_seg_image_pyramid_builder):
    mock_seg_image_pyramid_builder.seg_image_pyramid_regex = IMAGE_PYRAMID_DIR
    try:
        mock_seg_image_pyramid_builder._add_segmentation_image(None)
    except FileNotFoundError as e:
        assert str(e) == f"Segmentation assay with uuid {mock_seg_image_pyramid_builder._uuid} has no matching files"


def test_filtered_images_no_regex(mock_seg_image_pyramid_builder):
    mock_seg_image_pyramid_builder.seg_image_pyramid_regex = None
    try:
        mock_seg_image_pyramid_builder._add_segmentation_image(None)
    except ValueError as e:
        assert str(e) == "seg_image_pyramid_regex is not set. Cannot find segmentation images."


def mock_get_found_images(regex, file_paths):
    raise ValueError("Simulated failure in get_found_images")


def test_runtime_error_in_add_segmentation_image(mock_seg_image_pyramid_builder):
    with patch('src.portal_visualization.builders.imaging_builders.get_found_images',
               side_effect=mock_get_found_images):
        mock_seg_image_pyramid_builder.seg_image_pyramid_regex = "image_pyramid"

        with pytest.raises(RuntimeError) as err:
            mock_seg_image_pyramid_builder._add_segmentation_image(None)

        assert "Error while searching for segmentation images" in str(err.value)
        assert "Simulated failure in get_found_images" in str(err.value)


def test_find_segmentation_images_runtime_error():
    with pytest.raises(RuntimeError) as e:
        try:
            raise FileNotFoundError("No files found in the directory")
        except Exception as err:
            raise RuntimeError(f"Error while searching for segmentation images: {err}")

    assert "Error while searching for segmentation images:" in str(e.value)
    assert "No files found in the directory" in str(e.value)


def test_get_found_images():
    file_paths = [
        "image_pyramid/sample.ome.tiff",
        "image_pyramid/sample_separate/sample.ome.tiff",
    ]
    regex = "image_pyramid"
    result = get_found_images(regex, file_paths)
    assert len(result) == 1
    assert result[0] == "image_pyramid/sample.ome.tiff"


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate fixtures")
    parser.add_argument("--input", required=True, type=Path, help="Input JSON path")

    args = parser.parse_args()
    entity = json.loads(args.input.read_text())
    Builder = get_view_config_builder(entity, get_entity)
    builder = Builder(entity, "groups_token", "https://example.com/")
    conf, cells = builder.get_conf_cells()

    print(yaml.dump(clean_cells(cells), default_style="|"))
