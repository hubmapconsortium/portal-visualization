import json
from pathlib import Path

import pytest

from hubmap_commons.type_client import TypeClient

from src.builder_factory import get_view_config_builder


entity_paths = list((Path(__file__).parent / 'fixtures').glob("*/*-entity.json"))
assert len(entity_paths) > 0


def get_assay(name):
    # This code could also be used in portal-ui.
    # search-api might skip the REST interface.
    type_client = TypeClient('https://search.api.hubmapconsortium.org')
    return type_client.getAssayType(name)


@pytest.mark.parametrize(
    "entity_path", entity_paths, ids=lambda path: f'{path.parent.name}/{path.name}')
def test_entity_to_vitessce_conf(entity_path):
    entity = json.loads(entity_path.read_text())
    Builder = get_view_config_builder(entity, get_assay)
    assert Builder.__name__ == entity_path.parent.name

    builder = Builder(entity, 'groups_token', 'https://example.com/')
    conf = builder.get_conf_cells().conf

    conf_expected_path = entity_path.parent / entity_path.name.replace('-entity', '-conf')
    conf_expected = json.loads(conf_expected_path.read_text())

    assert conf_expected == conf
