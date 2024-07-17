import json
from flask import Flask
import pytest

from src.portal_visualization.client import ApiClient

mock_hit_source = {
    "uuid": "ABC123",
    "hubmap_id": "HMB123.XYZ",
    "mapped_metadata": {"age_unit": ["eons"], "age_value": [42]},
}

mock_es = {
    "hits": {
        "total": {"value": 1},
        "hits": [{"_id": "ABC123", "_source": mock_hit_source}],
    }
}


@pytest.fixture()
def app():
    app = Flask("test")
    app.config.update(
        {
            "TESTING": True,
            "ELASTICSEARCH_ENDPOINT": "search-api-url",
            "PORTAL_INDEX_PATH": "/",
        }
    )
    yield app


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

        def raise_for_status(self):
            pass

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
        "total": {"value": 10000},
        "hits": [{"_id": f"ABC{i}", "_source": mock_hit_source} for i in range(10000)],
    }
}


def mock_es_post_more_than_10k(path, **kwargs):
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self.text = "Logger call requires this"

        def json(self):
            return mock_es_more_than_10k

        def raise_for_status(self):
            pass

    return MockResponse()


def test_get_dataset_uuids_more_than_10k(app, mocker):
    mocker.patch("requests.post", side_effect=mock_es_post_more_than_10k)
    with app.app_context():
        api_client = ApiClient()
        with pytest.raises(Exception) as error_info:
            api_client.get_all_dataset_uuids()
            assert error_info.match("At least 10k datasets")


def test_get_entities(app, mocker):
    pass


def test_get_entity(app, mocker):
    pass


def test_get_latest_entity_uuid(app, mocker):
    pass


def test_get_files(app, mocker):
    pass


def test_get_vitessce_conf_cells_and_lifted_uuid(app, mocker):
    pass


def test_get_publication_ancillary_json(app, mocker):
    pass


def test_get_metadata_descriptions(app, mocker):
    pass
