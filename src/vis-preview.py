#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
from webbrowser import open_new_tab
from urllib.parse import quote_plus
from sys import stderr

import requests

from portal_visualization.builder_factory import get_view_config_builder
from portal_visualization.epic_factory import get_epic_builder
defaults = json.load((Path(__file__).parent / 'defaults.json').open())


def main():  # pragma: no cover
    assets_default_url = defaults['assets_url']

    parser = argparse.ArgumentParser(description='''
        Given HuBMAP Dataset JSON, generate a Vitessce viewconf, and load vitessce.io.''')
    input = parser.add_mutually_exclusive_group(required=True)
    input.add_argument(
        '--url', help='URL which returns Dataset JSON')
    input.add_argument(
        '--json', type=Path, help='File containing Dataset JSON')

    parser.add_argument(
        '--assets_url', metavar='URL',
        help=f'Assets endpoint; default: {assets_default_url}',
        default=assets_default_url)
    parser.add_argument(
        '--token', help='Globus groups token; Only needed if data is not public',
        default='')
    parser.add_argument(
        '--marker',
        help='Marker to highlight in visualization; Only used in some visualizations.')
    parser.add_argument(
        '--to_json', action='store_true',
        help='Output viewconf, rather than open in browser.')
    parser.add_argument(
        '--epic_uuid', metavar='UUID',
        help='uuid of the EPIC dataset.',
        default=None)
    parser.add_argument(
        '--parent_uuid', metavar='UUID',
        help='Parent uuid - Only needed for an image-pyramid support dataset.',
        default=None)

    args = parser.parse_args()
    marker = args.marker
    epic_uuid = args.epic_uuid
    parent_uuid = args.parent_uuid

    headers = get_headers(args.token)
    entity = get_entity_from_args(args.url, args.json, headers)

    Builder = get_view_config_builder(entity, get_entity, parent_uuid, epic_uuid)
    builder = Builder(entity, args.token, args.assets_url)
    print(f'Using: {builder.__class__.__name__}', file=stderr)
    conf_cells = builder.get_conf_cells(marker=marker)

    if (epic_uuid is not None and conf_cells is not None):  # pragma: no cover
        EpicBuilder = get_epic_builder(epic_uuid)
        epic_builder = EpicBuilder(epic_uuid, conf_cells, entity, args.token, args.assets_url)
        print(f'Using: {epic_builder.__class__.__name__}', file=stderr)
        conf_cells = epic_builder.get_conf_cells()

    if isinstance(conf_cells.conf, list):
        conf_as_json = json.dumps(conf_cells.conf[0])
    else:
        conf_as_json = json.dumps(conf_cells.conf)

    if args.to_json:
        print(conf_as_json)

    # For testing
    # with open('conf.json', 'w') as file:
    #     if isinstance(conf_cells.conf, list):
    #         json.dump(conf_cells.conf[0], file, indent=4, separators=(',', ': '))
    #     else:
    #         json.dump(conf_cells.conf, file, indent=4, separators=(',', ': '))

    data_url = f'data:,{quote_plus(conf_as_json)}'
    vitessce_url = f'http://vitessce.io/#?url={data_url}'
    open_new_tab(vitessce_url)


def get_headers(token):  # pragma: no cover
    global headers
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers


def get_entity(uuid):
    try:
        response = requests.get(f'{defaults["dataset_url"]}{uuid}.json', headers=headers)
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
        else:
            try:
                data = response.json()
                return data
            except Exception as e:
                print(f"Error in parsing the response {str(e)}")
    except Exception as e:
        print(f"Error accessing {defaults['assaytypes_url']}{uuid}: {str(e)}")


def get_entity_from_args(url_arg, json_arg, headers):  # pragma: no cover
    if url_arg:
        response = requests.get(url_arg, headers=headers)
        if response.status_code == 403:
            raise Exception('Protected data: Download JSON via browser; Redo with --json')
        response.raise_for_status()
        json_str = response.text
    else:
        json_str = json_arg.read_text()
    entity = json.loads(json_str)
    return entity


if __name__ == "__main__":  # pragma: no cover
    main()
