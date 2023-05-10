from vitessce import VitessceConfig

from .base_builders import ViewConfBuilder
from ..utils import get_conf_cells


class IntegrativeViewConfBuilder(ViewConfBuilder):
    # TODO: Add a test fixture.
    # TODO: Determine the name of the property that will hold the viewconfs.
    # TODO: If https://github.com/hubmapconsortium/portal-visualization/pull/66
    #       arrives first this can be replaced with:
    # def get_configs(self):
    #     return [
    #         VitessceConfig.from_dict(conf)
    #         for conf in self._entity['viewconfs']
    #     ]
    def get_conf_cells(self):
        return get_conf_cells(self._entity['viewconfs'])
