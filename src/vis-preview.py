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

def main():  # pragma: no cover
    defaults = json.load((Path(__file__).parent / 'defaults.json').open())
    assets_default_url = defaults['assets_url']
    assaytypes_default_url = defaults['assaytypes_url']

    parser = argparse.ArgumentParser(description='''
        Given HuBMAP Dataset JSON, generate a Vitessce viewconf, and load vitessce.io.''')
    input = parser.add_mutually_exclusive_group(required=True)
    input.add_argument(
        '--url', help='URL which returns Dataset JSON')
    input.add_argument(
        '--json', type=Path, help='File containing Dataset JSON')

    parser.add_argument(
        '--assaytypes_url', metavar='URL',
        help=f'AssayType service; default: {assaytypes_default_url}',
        default=assaytypes_default_url)
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
        '--parent_uuid', action='store_true',
        help='Parent uuid for the dataset',
        default=None)
    
    parser.add_argument(
        '--epic_uuid', metavar='URL',
        help='Epic dataset"s uuid',
        default=None)
    
    parser.add_argument(
        '--epic_builder',  action='store_true',
        help='Whether to use the epic_builder or not',
        default=None)

    args = parser.parse_args()
    marker = args.marker
    epic_builder = args.epic_builder
    epic_uuid = args.epic_uuid
    parent_uuid = args.parent_uuid # this may not be needed, as the --url provides the parent dataset json?

    if args.url:
        response = requests.get(args.url)
        if response.status_code == 403:
            raise Exception('Protected data: Download JSON via browser; Redo with --json')
        response.raise_for_status()
        json_str = response.text
    else:
        json_str = args.json.read_text()
    entity = json.loads(json_str)

    def get_assaytype(uuid):
        headers = {}
        if args.token:
            headers['Authorization'] = f'Bearer {args.token}'
        try:
            response = requests.get(f'{defaults["assaytypes_url"]}{uuid}', headers=headers)
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
            else:
                try:
                    data = response.json()
                    return data
                except Exception as e:
                    print("Error in parsing the response {str(e)}")
        except Exception as e:
            print(f"Error accessing {defaults['assaytypes_url']}{uuid}: {str(e)}")
    
    
    Builder = get_view_config_builder(entity, get_assaytype)
    builder = Builder(entity, args.token, args.assets_url)
    print(f'Using: {builder.__class__.__name__}', file=stderr)
    conf_cells = builder.get_conf_cells(marker=marker)
    
    if(epic_uuid is not None and conf_cells is not None):    
        EpicBuilder = get_epic_builder(conf_cells, epic_uuid)        
        epic_builder = EpicBuilder(conf_cells, epic_uuid, entity, args.token, args.assets_url)
        print(f'Using: {epic_builder.__class__.__name__}', file=stderr)
        conf_cells = epic_builder.get_conf_cells()


    if isinstance(conf_cells.conf, list):
        conf_as_json = json.dumps(conf_cells.conf[0])
    else:
        conf_as_json = json.dumps(conf_cells.conf)
        
    if args.to_json:
        print(conf_as_json)
        
    # with open ('epic.json','w') as file:
    #     if isinstance(conf_cells.conf, list):
    #         json.dump( conf_cells.conf[0], file, indent=4, separators=(',', ': '))
    #     else:
    #         json.dump( conf_cells.conf, file, indent=4, separators=(',', ': '))
            
            
    data_url = f'data:,{quote_plus(conf_as_json)}'
    vitessce_url = f'http://vitessce.io/#?url={data_url}'
    open_new_tab(vitessce_url)


if __name__ == "__main__":  # pragma: no cover
    main()
