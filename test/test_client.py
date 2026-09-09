import json

import pytest

try:
    from flask import Flask

    from portal_visualization.builders.base_builders import ConfCells
    from src.portal_visualization.client import ApiClient, _create_vitessce_error

    FULL_DEPS_AVAILABLE = True
except ImportError:
    FULL_DEPS_AVAILABLE = False
    # Skip entire module during collection if full dependencies not available
    pytest.skip("requires [full] optional dependencies", allow_module_level=True)

# Mark all tests in this file as requiring [full] dependencies
pytestmark = pytest.mark.requires_full

mock_hit_source = {
    "uuid": "ABC123",
    "hubmap_id": "HMB123.XYZ",
    "mapped_metadata": {"age_unit": ["eons"], "age_value": ["42"]},
}

flattened_hit_source = {
    "age_unit": "eons",
    "age_value": "42",
}

mock_es = {
    "hits": {
        "total": {"value": 1},
        "hits": [{"_id": "ABC123", "_source": mock_hit_source}],
    }
}


@pytest.fixture
def app():
    app = Flask("test")
    app.config.update(
        {
            "TESTING": True,
            "ELASTICSEARCH_ENDPOINT": "search-api-url",
            "PORTAL_INDEX_PATH": "/",
        }
    )
    return app


def mock_post_303(path, **kwargs):
    class MockResponse:
        def __init__(self):
            self.status_code = 303
            self.content = "s3-bucket-url"

        def raise_for_status(self):
            pass

    return MockResponse()


def mock_get_s3_json_file(path, **kwargs):
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self.content = json.dumps(mock_es)
            self.text = json.dumps(mock_es)

        def raise_for_status(self):
            pass

        def json(self):
            return mock_es

    return MockResponse()


def test_s3_redirect(mocker):
    mocker.patch("requests.post", side_effect=mock_post_303)
    mocker.patch("requests.get", side_effect=mock_get_s3_json_file)
    api_client = ApiClient()
    response = api_client._request("search-api-url", body_json={"query": {}})
    assert response == mock_es


def mock_es_post(path, **kwargs):
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self.text = "Logger call requires this"

        def json(self):
            return mock_es

        def raise_for_status(self):
            pass

    return MockResponse()


def test_get_descendant_to_lift(app, mocker):
    mocker.patch("requests.post", side_effect=mock_es_post)
    with app.app_context():
        api_client = ApiClient()
        descendant = api_client.get_descendant_to_lift("uuid123")
    assert descendant == mock_hit_source


def mock_es_post_no_hits(path, **kwargs):
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self.text = "Logger call requires this"

        def json(self):
            return {"hits": {"total": {"value": 0}, "hits": []}}

        def raise_for_status(self):
            pass

    return MockResponse()


def test_get_descendant_to_lift_error(app, mocker):
    mocker.patch("requests.post", side_effect=mock_es_post_no_hits)
    with app.app_context():
        api_client = ApiClient()
        descendant = api_client.get_descendant_to_lift("uuid123")
    assert descendant is None


def test_get_descendant_to_lift_status_filter_includes_approval(app, mocker):
    """Image pyramid lifting must consider QA, Approval, and Published descendants."""
    captured = {}

    def side_effect(path, **kwargs):
        captured["body"] = kwargs.get("json")
        return mock_es_post(path, **kwargs)

    mocker.patch("requests.post", side_effect=side_effect)
    with app.app_context():
        api_client = ApiClient()
        api_client.get_descendant_to_lift("uuid123")

    status_terms = next(
        clause["terms"]["mapped_status.keyword"]
        for clause in captured["body"]["query"]["bool"]["must"]
        if "terms" in clause and "mapped_status.keyword" in clause["terms"]
    )
    assert set(status_terms) == {"QA", "Approval", "Published"}


def test_clean_headers(app):
    test_headers = {
        "Authorization": "Bearer token",
        "Content-Type": "application/json",
        "X-Test": "test",
    }
    with app.app_context():
        api_client = ApiClient()
        cleaned_headers = api_client._clean_headers(test_headers)
        assert cleaned_headers == {
            "Authorization": "REDACTED",
            "Content-Type": "application/json",
            "X-Test": "test",
        }


def test_get_all_dataset_uuids(app, mocker):
    mocker.patch("requests.post", side_effect=mock_es_post)
    with app.app_context():
        api_client = ApiClient()
        uuids = api_client.get_all_dataset_uuids()
    assert uuids == ["ABC123"]


mock_es_more_than_10k = {
    "hits": {
        "total": {"value": 10001},
        "hits": [{"_id": f"ABC{i}", "_source": mock_hit_source, "sort": [f"ABC{i}"]} for i in range(10000)],
    }
}

mock_es_page_2 = {
    "hits": {
        "total": {"value": 10001},
        "hits": [{"_id": "ABC10000", "_source": mock_hit_source, "sort": ["ABC10000"]}],
    }
}


def _mock_es_post_paginated():
    """Returns a side_effect function that returns page 1 then page 2."""
    call_count = 0

    def side_effect(path, **kwargs):
        nonlocal call_count
        call_count += 1
        data = mock_es_more_than_10k if call_count == 1 else mock_es_page_2

        class MockResponse:
            def __init__(self):
                self.status_code = 200
                self.text = "Logger call requires this"

            def json(self):
                return data

            def raise_for_status(self):
                pass

        return MockResponse()

    return side_effect


def test_get_dataset_uuids_more_than_10k(app, mocker):
    mocker.patch("requests.post", side_effect=_mock_es_post_paginated())
    with app.app_context():
        api_client = ApiClient()
        uuids = api_client.get_all_dataset_uuids()
    assert len(uuids) == 10001
    assert uuids[0] == "ABC0"
    assert uuids[-1] == "ABC10000"


@pytest.mark.parametrize("plural_lc_entity_type", ["datasets", "samples", "donors"])
def test_get_entities(app, mocker, plural_lc_entity_type):
    mocker.patch("requests.post", side_effect=mock_es_post)
    with app.app_context():
        api_client = ApiClient()
        entities = api_client.get_entities(plural_lc_entity_type)
        assert json.dumps(entities, indent=2) == json.dumps([flattened_hit_source], indent=2)


def test_get_entities_paginates(app, mocker):
    mocker.patch("requests.post", side_effect=_mock_es_post_paginated())
    with app.app_context():
        api_client = ApiClient()
        entities = api_client.get_entities("datasets")
        assert len(entities) == 10001


def test_get_entities_pagination_empty_page(app, mocker):
    """Cover the `if not hits: break` branch when a page returns no results."""
    mock_es_empty_page = {"hits": {"total": {"value": 10001}, "hits": []}}
    call_count = 0

    def side_effect(path, **kwargs):
        nonlocal call_count
        call_count += 1
        data = mock_es_more_than_10k if call_count == 1 else mock_es_empty_page

        class MockResponse:
            def __init__(self):
                self.status_code = 200
                self.text = "Logger call requires this"

            def json(self):
                return data

            def raise_for_status(self):
                pass

        return MockResponse()

    mocker.patch("requests.post", side_effect=side_effect)
    with app.app_context():
        api_client = ApiClient()
        entities = api_client.get_entities("datasets")
        assert len(entities) == 10000


def test_get_entities_with_post_filter_extra(app, mocker):
    mocker.patch("requests.post", side_effect=mock_es_post)
    with app.app_context():
        api_client = ApiClient()
        entities = api_client.get_entities(
            "samples",
            post_filter_extra={"exists": {"field": "descendant_counts.entity_type.Dataset"}},
        )
        assert len(entities) == 1


def test_get_entities_post_filter_extra_merges_bool_clause(app, mocker):
    """A `bool` post_filter_extra should have its `should`/`must_not` clauses preserved
    and its `must` array merged with the entity_type filter (not nested)."""
    captured = {}

    def side_effect(path, **kwargs):
        captured["body"] = kwargs.get("json")

        class MockResponse:
            status_code = 200
            text = "Logger call requires this"

            def json(self):
                return mock_es

            def raise_for_status(self):
                pass

        return MockResponse()

    mocker.patch("requests.post", side_effect=side_effect)
    with app.app_context():
        api_client = ApiClient()
        api_client.get_entities(
            "samples",
            post_filter_extra={
                "bool": {
                    "must": [{"term": {"a": "1"}}],
                    "should": [{"term": {"b": "2"}}],
                    "must_not": [{"term": {"c": "3"}}],
                }
            },
        )

    post_filter = captured["body"]["post_filter"]
    assert post_filter["bool"]["must"] == [
        {"term": {"entity_type.keyword": "Sample"}},
        {"term": {"a": "1"}},
    ]
    assert post_filter["bool"]["should"] == [{"term": {"b": "2"}}]
    assert post_filter["bool"]["must_not"] == [{"term": {"c": "3"}}]


def test_get_entities_pagination_safety_limit(app, mocker):
    """If total_hits keeps claiming more results than pages deliver, the loop must
    bail out instead of spinning forever."""
    single_hit_page = {
        "hits": {
            "total": {"value": 10_000_000},
            "hits": [{"_id": "ABC", "_source": mock_hit_source, "sort": ["ABC"]}],
        }
    }

    def side_effect(path, **kwargs):
        class MockResponse:
            status_code = 200
            text = "Logger call requires this"

            def json(self):
                return single_hit_page

            def raise_for_status(self):
                pass

        return MockResponse()

    mocker.patch("requests.post", side_effect=side_effect)
    with app.app_context():
        api_client = ApiClient()
        with pytest.raises(Exception, match="Pagination safety limit"):
            api_client.get_entities("datasets")


@pytest.mark.parametrize("params", [{"uuid": "uuid"}, {"hbm_id": "hubmap_id"}])
def test_get_entity(app, mocker, params):
    mocker.patch("requests.post", side_effect=mock_es_post)
    with app.app_context():
        api_client = ApiClient()
        entity = api_client.get_entity(**params)
        assert json.dumps(entity, indent=2) == json.dumps(mock_hit_source, indent=2)


def test_get_entity_source_exclude(app, mocker):
    """The heavy nested relative lists are opt-in to drop, so the default query stays whole-doc
    (the portal's ``.json`` route exists to surface it) and the exclusion only appears when asked.
    """
    from src.portal_visualization.client import HEAVY_RELATIVE_FIELDS

    bodies = []

    def capture(path, **kwargs):
        bodies.append(kwargs.get("json"))
        return mock_es_post(path, **kwargs)

    mocker.patch("requests.post", side_effect=capture)
    with app.app_context():
        api_client = ApiClient()
        api_client.get_entity(uuid="uuid")
        api_client.get_entity(uuid="uuid", source_exclude=HEAVY_RELATIVE_FIELDS)

    assert "_source" not in bodies[0]
    assert bodies[1]["_source"] == {"exclude": HEAVY_RELATIVE_FIELDS}
    # donor (singular) and the id lists are what the portal actually reads; they must survive.
    for kept in ("donor", "ancestor_ids", "descendant_ids", "files", "metadata"):
        assert kept not in HEAVY_RELATIVE_FIELDS


def test_get_descendant_to_lift_excludes_relatives(app, mocker):
    """The lifted document is handed to a builder as an entity, so only the nested relative
    lists may be dropped -- files/metadata/hints must stay."""
    from src.portal_visualization.client import HEAVY_RELATIVE_FIELDS

    bodies = []

    def capture(path, **kwargs):
        bodies.append(kwargs.get("json"))
        return mock_es_post(path, **kwargs)

    mocker.patch("requests.post", side_effect=capture)
    with app.app_context():
        api_client = ApiClient()
        api_client.get_descendant_to_lift("uuid123")

    assert bodies[0]["_source"] == {"exclude": HEAVY_RELATIVE_FIELDS}


def test_get_entity_two_ids(app, mocker):
    with app.app_context():
        api_client = ApiClient()
        with pytest.raises(Exception) as error_info:  # noqa: PT011, PT012
            api_client.get_entity(uuid="uuid", hbm_id="hubmap_id")
            assert error_info.match("Only UUID or HBM ID should be provided")  # pragma: no cover


def mock_get_revisions(path, **kwargs):
    mock_revisions = [
        {"uuid": "ABC123", "revision_number": 10},
        {"uuid": "DEF456", "revision_number": 11},
    ]

    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self.text = "Logger call requires this"
            self.content = json.dumps(mock_revisions)

        def json(self):
            return mock_revisions

        def raise_for_status(self):
            pass

    return MockResponse()


@pytest.mark.parametrize(
    "params",
    [
        {"uuid": "uuid", "type": "dataset"},
        {"uuid": "uuid", "type": "sample"},
        {"uuid": "uuid", "type": "donor"},
    ],
)
def test_get_latest_entity_uuid(app, mocker, params):
    mocker.patch("requests.get", side_effect=mock_get_revisions)
    with app.app_context():
        api_client = ApiClient(entity_api_endpoint="entity-api-url")
        entity_uuid = api_client.get_latest_entity_uuid(**params)
        assert entity_uuid == "DEF456"


def mock_files_response(path, **kwargs):
    mock_file_response = {
        "hits": {
            "hits": [
                {"_id": "1234", "_source": {"files": [{"rel_path": "abc.txt"}]}},
                {"_id": "5678", "_source": {"files": [{"rel_path": "def.txt"}]}},
            ]
        }
    }

    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self.text = "Logger call requires this"
            self.content = json.dumps(mock_file_response)

        def json(self):
            return mock_file_response

        def raise_for_status(self):
            pass

    return MockResponse()


def test_get_files(app, mocker):
    mocker.patch("requests.post", side_effect=mock_files_response)
    with app.app_context():
        api_client = ApiClient()
        files = api_client.get_files(["1234", "5678"])
        assert files == {"1234": ["abc.txt"], "5678": ["def.txt"]}


related_entity_no_files_error = _create_vitessce_error(
    'Related image entity ABC123 is missing file information (no "files" key found in its metadata).'
)


@pytest.mark.parametrize(
    ("entity", "patched_function", "side_effect", "expected_conf", "expected_vis_lifted_uuid"),
    [
        (
            # No metadata in descendant
            {"uuid": "12345"},
            "requests.post",
            mock_es_post,
            related_entity_no_files_error,
            None,
        ),
        (
            # No descendants, not marked as having a visualization, no files
            {"uuid": "12345"},
            "requests.post",
            mock_es_post_no_hits,
            ConfCells(None, None),
            None,
        ),
        # TODO? Add more test cases for happy scenarios
    ],
)
def test_get_vitessce_conf_cells_and_lifted_uuid(
    app,
    mocker,
    entity,
    patched_function,
    side_effect,
    expected_conf,
    expected_vis_lifted_uuid,
):
    mocker.patch(patched_function, side_effect=side_effect)
    with app.app_context():
        api_client = ApiClient(
            groups_token="token",
            elasticsearch_endpoint="http://example.com",
            portal_index_path="/",
            ubkg_endpoint="http://example.com",
            entity_api_endpoint="http://example.com",
            soft_assay_endpoint="http://example.com",
            soft_assay_endpoint_path="/",
        )
        vitessce_conf = api_client.get_vitessce_conf_cells_and_lifted_uuid(entity)
        assert vitessce_conf.vitessce_conf == expected_conf
        assert vitessce_conf.vis_lifted_uuid == expected_vis_lifted_uuid


@pytest.mark.parametrize("groups_token", [None, "token"])
def test_get_publication_ancillary_json(app, mocker, groups_token):
    mocker.patch("requests.post", side_effect=mock_es_post)
    mocker.patch("requests.get", side_effect=mock_get_s3_json_file)
    with app.app_context():
        api_client = ApiClient(groups_token=groups_token)
        result = api_client.get_publication_ancillary_json({"uuid": "ABC123"})
        assert result.publication_json == mock_es
        assert result.vis_lifted_uuid == "ABC123"


def test_get_metadata_descriptions(app, mocker):
    mocker.patch("requests.get", side_effect=mock_get_s3_json_file)
    with app.app_context():
        api_client = ApiClient()
        metadata_descriptions = api_client.get_metadata_descriptions()
        assert metadata_descriptions == mock_es


def test_get_metadata_field_types(app, mocker):
    mocker.patch("requests.get", side_effect=mock_get_s3_json_file)
    with app.app_context():
        api_client = ApiClient()
        metadata_types = api_client.get_metadata_field_types()
        assert metadata_types == mock_es
