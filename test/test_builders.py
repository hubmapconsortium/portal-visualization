#!/usr/bin/env python3
import argparse
import yaml
import json
from pathlib import Path
from os import environ
from dataclasses import dataclass

import pytest
import zarr

from src.portal_visualization.utils import get_found_images, read_zip_zarr
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
    if not isinstance(input, str):
        uuid = input.get("uuid")
    else:
        uuid = input
    if uuid is None:  # pragma: no cover
        return default_assaytype
    assay = json.loads(assaytypes_path.joinpath(f"{uuid}.json").read_text())
    return assay


# Construct test cases for has_visualization.
# Initial values are edge cases (null view conf builder)
has_visualization_test_cases = [
    (False, {"uuid": "2c2179ea741d3bbb47772172a316a2bf"}),
    (False, {"uuid": "f9ae931b8b49252f150d7f8bf1d2d13f-bad"}),
]
excluded_uuids = {entity["uuid"] for _, entity in has_visualization_test_cases}

# All other values are good entities which should have a visualization
for path in good_entity_paths:
    entity = json.loads(path.read_text())
    uuid = entity.get("uuid")
    if uuid in excluded_uuids or path.parent.name == "NullViewConfBuilder":
        continue
    has_visualization_test_cases.append((True, entity))


@pytest.mark.parametrize(
    "has_vis_entity",
    has_visualization_test_cases,
    ids=lambda e: (
        f"has_visualization={e[0]}_uuid={e[1].get('uuid', 'no-uuid')}"
        if isinstance(e, tuple) else str(e)
    )
)
def test_has_visualization(has_vis_entity):
    has_vis, entity = has_vis_entity
    parent = entity.get("parent") or None  # Only used for image pyramids
    hints = entity.get("vitessce-hints", [])
    epic_uuid = (  # For segmentation masks
        entity.get("uuid")
        if "epic" in hints and len(hints) > 1
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
        elif "pan-az" in entity_path.name:
            z["obs/azimuth_broad"] = True  # only checked for membership in zarr group
            z["obs/azimuth_medium"] = True
            z["obs/azimuth_fine"] = True
            z["obs/CL_Label"] = True
            z["obs/final_level_labels"] = True
            z["obs/full_hierarchical_labels"] = True
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
    if "object-by-analyte" in entity_path.name:
        entity = json.loads(entity_path.read_text())
        # Mock the HTTP request that would fetch the metadata
        mock_response = mocker.Mock()
        mock_response.json.return_value = entity.get("secondary_analysis_metadata")
        mock_response.raise_for_status.return_value = None
        mocker.patch("requests.get", return_value=mock_response)
    mocker.patch("zarr.open", return_value=z)


def test_read_zip_zarr_opens_store(mocker):
    # Mock the fsspec filesystem and zarr open
    mock_fs = mocker.Mock()
    mock_mapper = mocker.Mock()
    mock_zarr_obj = mocker.Mock()

    mock_fs.get_mapper.return_value = mock_mapper

    mocker.patch("src.portal_visualization.utils.fsspec.filesystem", return_value=mock_fs)
    mocker.patch("src.portal_visualization.utils.zarr.open", return_value=mock_zarr_obj)

    dummy_url = "https://example.com/fake.zarr.zip"
    request_init = {"headers": {"Authorization": "Bearer token"}}

    result = read_zip_zarr(dummy_url, request_init)

    assert result == mock_zarr_obj
    mock_fs.get_mapper.assert_called_once_with("")


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
    if "zip" in str(entity_path):
        mock_zarr = mocker.Mock()
        mocker.patch("src.portal_visualization.utils.read_zip_zarr", return_value=mock_zarr)

    is_object_by_analyte = "epic" in assay_type["vitessce-hints"] and len(
        assay_type["vitessce-hints"]) == 1

    # If "epic" is the only hint, it's object by analyte and doesn't need a parent UUID
    if "epic" in assay_type["vitessce-hints"] and not is_object_by_analyte:
        epic_uuid = entity.get("uuid")

    Builder = get_view_config_builder(entity, get_entity, parent, epic_uuid)
    # Envvars should not be set during normal test runs,
    # but to test the end-to-end integration, they are useful.
    # epic_uuid = environ.get("EPIC_UUID", "epic_uuid")
    builder = Builder(entity, groups_token, assets_url)
    conf, cells = builder.get_conf_cells(marker=marker)

    # Uncomment to generate a fixture
    print(json.dumps(conf, indent=2))

    if "epic" not in assay_type["vitessce-hints"] or is_object_by_analyte:
        assert Builder.__name__ == entity_path.parent.name
        compare_confs(entity_path, conf, cells)
    elif "epic" in assay_type["vitessce-hints"]:
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


def test_get_found_images_error_handling():
    file_paths = [
        "image_pyramid/sample.ome.tiff",
        "image_pyramid/sample_separate/sample.ome.tiff",
    ]
    regex = "["  # invalid regex, forces re.error

    with pytest.raises(RuntimeError) as excinfo:
        try:
            get_found_images(regex, file_paths)
        except Exception as e:
            raise RuntimeError(f"Error while searching for pyramid images: {e}")

    assert "Error while searching for pyramid images" in str(excinfo.value)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate fixtures")
    parser.add_argument("--input", required=True, type=Path, help="Input JSON path")

    args = parser.parse_args()
    entity = json.loads(args.input.read_text())
    Builder = get_view_config_builder(entity, get_entity)
    builder = Builder(entity, "groups_token", "https://example.com/")
    conf, cells = builder.get_conf_cells()

    print(yaml.dump(clean_cells(cells), default_style="|"))
