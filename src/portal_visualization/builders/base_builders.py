import urllib
from abc import ABC, abstractmethod
from collections import namedtuple

ConfCells = namedtuple("ConfCells", ["conf", "cells"])


class NullViewConfBuilder:
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        # Just so it has the same signature as the other builders
        pass

    def get_conf_cells(self, **kwargs):
        return ConfCells(None, None)


class ViewConfBuilder(ABC):
    def __init__(self, entity, groups_token, assets_endpoint, **kwargs):
        """Object for building the vitessce configuration.
        :param dict entity: Entity response from search index (from the entity API)
        :param str  groups_token: Groups token for use in authenticating API
        :param str  assets_endpoint: The base URL for the assets API
        :param dict kwargs: Additional keyword arguments
        :param str  kwargs.schema_version: The vitessce schema version to use, default "1.0.15"
        :param bool kwargs.minimal: Whether or not to build a minimal configuration, default False
        """

        self._uuid = entity["uuid"]
        self._groups_token = groups_token
        self._assets_endpoint = assets_endpoint
        self._entity = entity
        self._files = []
        self._schema_version = kwargs.get("schema_version", "1.0.15")
        self._minimal = kwargs.get("minimal", False)

    @abstractmethod
    def get_conf_cells(self, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def _replace_url_in_file(self, file):
        """Replace url in incoming file object
        :param dict file: File dict which will have its rel_path replaced by url
        :rtype: dict The file with rel_path replaced by url

        >>> from pprint import pprint
        >>> builder = _DocTestBuilder(
        ...   entity={ "uuid": "uuid" },
        ...   groups_token='groups_token',
        ...   assets_endpoint='https://example.com')
        >>> file = {
        ...     'file_type': 'cells.json',
        ...     'rel_path': 'cells.json',
        ...     'coordination_values': { 'obsType': 'cell' } }
        >>> pprint(builder._replace_url_in_file(file))
        {'coordination_values': {'obsType': 'cell'},
         'file_type': 'cells.json',
         'url': 'https://example.com/uuid/cells.json?token=groups_token'}
        """

        return {
            "coordination_values": file["coordination_values"],
            "file_type": file["file_type"],
            "url": self._build_assets_url(file["rel_path"]),
        }

    def _build_assets_url(self, rel_path, use_token=True):
        """Create a url for an asset.
        :param str rel_path: The path off of which the url should be built
        :param bool use_token: Whether or not to append a groups token to the URL, default True
        :rtype: dict The file with rel_path replaced by url

        >>> from pprint import pprint
        >>> builder = _DocTestBuilder(
        ...   entity={ "uuid": "uuid" },
        ...   groups_token='groups_token',
        ...   assets_endpoint='https://example.com')
        >>> builder._build_assets_url("rel_path/to/clusters.ome.tiff")
        'https://example.com/uuid/rel_path/to/clusters.ome.tiff?token=groups_token'

        """
        uuid = self._uuid
        if hasattr(self, "_epic_uuid"):  # pragma: no cover
            uuid = self._epic_uuid
        base_url = urllib.parse.urljoin(self._assets_endpoint, f"{uuid}/{rel_path}")
        token_param = urllib.parse.urlencode({"token": self._groups_token})
        return f"{base_url}?{token_param}" if use_token else base_url

    def _get_request_init(self):
        """Get request headers for requestInit parameter in Vitessce conf.
        This is needed for non-public zarr stores because the client forms URLs for zarr chunks,
        not the above _build_assets_url function.

        >>> builder = _DocTestBuilder(
        ...   entity={"uuid": "uuid", "status": "QA"},
        ...   groups_token='groups_token',
        ...   assets_endpoint='https://example.com')
        >>> builder._get_request_init()
        {'headers': {'Authorization': 'Bearer groups_token'}}

        >>> builder = _DocTestBuilder(
        ...   entity={"uuid": "uuid", "status": "Published"},
        ...   groups_token='groups_token',
        ...   assets_endpoint='https://example.com')
        >>> repr(builder._get_request_init())
        'None'
        """
        if self._entity["status"] == "Published":
            # Extra headers outside of a select few cause extra CORS-preflight requests which
            # can slow down the webpage.  If the dataset is published, we don't need to use
            # header to authenticate access to the assets API.
            # See: https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS#simple_requests
            return None
        return {"headers": {"Authorization": f"Bearer {self._groups_token}"}}

    def _get_file_paths(self):
        """Get all rel_path keys from the entity dict.

        >>> files = [{ "rel_path": "path/to/file" }, { "rel_path": "path/to/other_file" }]
        >>> builder = _DocTestBuilder(
        ...   entity={"uuid": "uuid", "files": files},
        ...   groups_token='groups_token',
        ...   assets_endpoint='https://example.com')
        >>> builder._get_file_paths()
        ['path/to/file', 'path/to/other_file']
        """
        return [file["rel_path"] for file in self._entity["files"]]


class _DocTestBuilder(ViewConfBuilder):  # pragma: no cover
    # The doctests on the methods in this file need a concrete class to instantiate:
    # We need a concrete definition for this method, even if it's never used.
    def get_conf_cells(self, **kwargs):
        pass
