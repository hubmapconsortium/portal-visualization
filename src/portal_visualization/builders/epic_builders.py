
from abc import ABC, abstractmethod
from vitessce import VitessceConfig
from .base_builders import ConfCells
from ..utils import get_conf_cells
# EPIC builders take in a vitessce conf output by a previous builder and modify it
# accordingly to add the EPIC-specific configuration.


class EPICConfBuilder(ABC):
    def __init__(self, base_conf: ConfCells) -> None:
        self._base_conf = VitessceConfig(base_conf[0])
        pass

    @abstractmethod
    def get_conf_cells(self, **kwargs):
        pass


class SegmentationMaskBuilder(EPICConfBuilder):
    def get_conf_cells(self, **kwargs):
        return get_conf_cells(self._base_conf)
