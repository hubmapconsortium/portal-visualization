#!/usr/bin/env python3
import argparse
import yaml
import json
from pathlib import Path
from os import environ
from dataclasses import dataclass

import pytest

from hubmap_commons.type_client import TypeClient

from src.builder_factory import get_view_config_builder, has_visualization


@dataclass
class MockResponse:
    content: str


good_entity_paths = list((Path(__file__).parent / 'good-fixtures').glob("*/*-entity.json"))
assert len(good_entity_paths) > 0

bad_entity_paths = list((Path(__file__).parent / 'bad-fixtures').glob("*-entity.json"))
assert len(bad_entity_paths) > 0


def get_assay(name):
    # This code could also be used in portal-ui.
    # search-api might skip the REST interface.
    type_client = TypeClient('https://search.api.hubmapconsortium.org')
    return type_client.getAssayType(name)


@pytest.mark.parametrize(
    "hasvis_entity",
    [
        (False, {'data_types': [], 'metadata': {'dag_provenance_list': []}}),
        (True, json.loads(Path.read_text(good_entity_paths[0])))
        # If the first fixture returns a Null builder this would break.
    ],
    ids=lambda hasvis_entity: f'has_visualization={hasvis_entity[0]}')
def test_has_visualization(hasvis_entity):
    hasvis, entity = hasvis_entity
    assert hasvis == has_visualization(entity, get_assay)


@pytest.mark.parametrize(
    "entity_path", good_entity_paths, ids=lambda path: f'{path.parent.name}/{path.name}')
def test_entity_to_vitessce_conf(entity_path, mocker):
    entity = json.loads(entity_path.read_text())
    Builder = get_view_config_builder(entity, get_assay)
    assert Builder.__name__ == entity_path.parent.name

    # Envvars should not be set during normal test runs,
    # but to test the end-to-end integration, they are useful.
    groups_token = environ.get('GROUPS_TOKEN', 'groups_token')
    assets_url = environ.get('ASSETS_URL', 'https://example.com')
    if 'ASSETS_URL' not in environ:
        mock_response = (
            MockResponse(content=b'\x01')
            if 'http200' in entity_path.name else
            MockResponse(content=b'something else')
        )
        mocker.patch('requests.get', return_value=mock_response)

    builder = Builder(entity, groups_token, assets_url)
    conf, cells = builder.get_conf_cells()

    expected_conf_path = entity_path.parent / entity_path.name.replace('-entity', '-conf')
    expected_conf = json.loads(expected_conf_path.read_text())
    assert expected_conf == conf

    expected_cells_path = (
        entity_path.parent / entity_path.name.replace('-entity.json', '-cells.yaml'))
    if expected_cells_path.is_file():
        expected_cells = yaml.safe_load(expected_cells_path.read_text())
        assert expected_cells == clean_cells(cells)


@pytest.mark.parametrize(
    "entity_path", bad_entity_paths, ids=lambda path: path.name)
def test_entity_to_error(entity_path):
    entity = json.loads(entity_path.read_text())
    with pytest.raises(Exception) as error_info:
        Builder = get_view_config_builder(entity, get_assay)
        builder = Builder(entity, 'groups_token', 'https://example.com/')
        builder.get_conf_cells()
    actual_error = f'{error_info.type.__name__}: {error_info.value.args[0]}'

    error_expected_path = (
        entity_path.parent / entity_path.name.replace('-entity.json', '-error.txt'))
    expected_error = error_expected_path.read_text().strip()
    assert actual_error == expected_error


def clean_cells(cells):
    return [
        {
            k: v for k, v in dict(c).items()
            if k not in {'metadata', 'id', 'execution_count', 'outputs'}
        } for c in cells
    ]


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(description='Generate fixtures')
    parser.add_argument(
        '--input', required=True, type=Path, help='Input JSON path')

    args = parser.parse_args()
    entity = json.loads(args.input.read_text())
    Builder = get_view_config_builder(entity, get_assay)
    builder = Builder(entity, 'groups_token', 'https://example.com/')
    conf, cells = builder.get_conf_cells()

    print(yaml.dump(clean_cells(cells), default_style='|'))
