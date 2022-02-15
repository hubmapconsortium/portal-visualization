import json
from pathlib import Path
from os import environ
from dataclasses import dataclass

import pytest

from hubmap_commons.type_client import TypeClient

from src.builder_factory import get_view_config_builder


@dataclass
class MockResponse:
    ok: bool
    status_code: int


entity_paths = list((Path(__file__).parent / 'fixtures').glob("*/*-entity.json"))
assert len(entity_paths) > 0


def get_assay(name):
    # This code could also be used in portal-ui.
    # search-api might skip the REST interface.
    type_client = TypeClient('https://search.api.hubmapconsortium.org')
    return type_client.getAssayType(name)


@pytest.mark.parametrize(
    "entity_path", entity_paths, ids=lambda path: f'{path.parent.name}/{path.name}')
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
            MockResponse(ok=True, status_code=200)
            if 'http200' in entity_path.name else
            MockResponse(ok=False, status_code=404)
        )
        mocker.patch('requests.get', return_value=mock_response)

    builder = Builder(entity, groups_token, assets_url)
    conf = builder.get_conf_cells().conf

    conf_expected_path = entity_path.parent / entity_path.name.replace('-entity', '-conf')
    conf_expected = json.loads(conf_expected_path.read_text())

    assert conf_expected == conf
