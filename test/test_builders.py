import json
from pathlib import Path
from os import environ
from dataclasses import dataclass

import pytest

from hubmap_commons.type_client import TypeClient

from src.builder_factory import get_view_config_builder


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
    conf = builder.get_conf_cells().conf

    expected_conf_path = entity_path.parent / entity_path.name.replace('-entity', '-conf')
    expected_conf = json.loads(expected_conf_path.read_text())
    assert expected_conf == conf


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
