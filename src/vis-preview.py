#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
from webbrowser import open_new_tab
from urllib.parse import quote_plus
from sys import stderr

import requests

from hubmap_commons.type_client import TypeClient

from portal_visualization.builder_factory import get_view_config_builder


def main():  # pragma: no cover
    types_default_url = 'https://search.api.hubmapconsortium.org'
    assets_default_url = 'https://assets.hubmapconsortium.org'

    parser = argparse.ArgumentParser(description='''
        Given HuBMAP Dataset JSON, generate a Vitessce viewconf, and load vitessce.io.''')
    input = parser.add_mutually_exclusive_group(required=True)
    input.add_argument(
        '--url', help='URL which returns Dataset JSON')
    input.add_argument(
        '--json', type=Path, help='File containing Dataset JSON')

    parser.add_argument(
        '--types_url', metavar='URL',
        help=f'Type service; default: {types_default_url}',
        default=types_default_url)
    parser.add_argument(
        '--assets_url', metavar='URL',
        help=f'Assets endpoint; default: {assets_default_url}',
        default=assets_default_url)
    parser.add_argument(
        '--token', help='Globus groups token; Only needed if data is not public',
        default='')
    parser.add_argument(
        '--marker_gene', metavar='GENE',
        help='Gene to highlight in visualization; Only used in some visualizations.')
    parser.add_argument(
        '--to_json', action='store_true',
        help='Output viewconf, rather than open in browser.')

    args = parser.parse_args()
    marker_gene = args.marker_gene

    if args.url:
        response = requests.get(args.url)
        if response.status_code == 403:
            # Even if the user has provided a globus token,
            # that isn't useful when making requests to our portal.
            raise Exception('Protected data: Download JSON via browser; Redo with --json')
        response.raise_for_status()
        json_str = response.text
    else:
        json_str = args.json.read_text()
    entity = json.loads(json_str)

    def get_assay(name):
        type_client = TypeClient(args.types_url)
        return type_client.getAssayType(name)

    Builder = get_view_config_builder(entity=entity, get_assay=get_assay)
    builder = Builder(entity, args.token, args.assets_url)
    print(f'Using: {builder.__class__.__name__}', file=stderr)
    conf_cells = builder.get_conf_cells(marker_gene=marker_gene)
    if args.to_json:
        print(json.dumps(conf_cells.conf, indent=2))
    else:
        conf_as_json = json.dumps(conf_cells.conf)
        data_url = f'data:,{quote_plus(conf_as_json)}'
        vitessce_url = f'http://vitessce.io/#?url={data_url}'
        open_new_tab(vitessce_url)


if __name__ == "__main__":  # pragma: no cover
    main()
