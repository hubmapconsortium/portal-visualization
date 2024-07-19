from abc import ABC, abstractmethod
from vitessce import VitessceConfig
from .base_builders import ConfCells
from ..utils import get_conf_cells


# EPIC builders take in a vitessce conf output by a previous builder and modify it
# accordingly to add the EPIC-specific configuration.
class EPICConfBuilder(ABC):
    def __init__(self, base_conf: ConfCells, epic_uuid) -> None:

        conf, cells = base_conf

        if conf is None:
            raise ValueError("ConfCells object must have a conf attribute")

        self._is_plural = isinstance(conf, list)

        if self._is_plural:
            self._base_conf = [
                VitessceConfig.from_dict(conf) for conf in conf
            ]
        else:
            self._base_conf: VitessceConfig = VitessceConfig.from_dict(base_conf.conf)

        self._epic_uuid = epic_uuid
        pass

    def get_conf_cells(self):
        self.apply()
        if (self._is_plural):
            return get_conf_cells([conf.to_dict() for conf in self._base_conf])
        return get_conf_cells(self._base_conf)

    def apply(self):
        if self._is_plural:
            for conf in self._base_conf:
                self._apply(conf)
        else:
            self._apply(self._base_conf)

    @abstractmethod
    def _apply(self, conf):  # pragma: no cover
        pass


class SegmentationMaskBuilder(EPICConfBuilder):

    def _apply(self, conf):
        datasets = conf.get_datasets()
        print(f"Found {len(datasets)} datasets")
        # Proof of concept using one of the kaggle segmentation masks for now
        # segmentations = ObsSegmentationsOmeTiffWrapper(
        #     img_url='https://assets.hubmapconsortium.org/c9d9ab5c9ee9642b60dd351024968627/ometiff-pyramids/VAN0042-RK-3-18-registered-PAS-to-postAF-registered.ome_mask.ome.tif?token=AgndN7NVbn83wwDXjpnY1Y0lDoJj2j7zOGmn1WN6qr9pqdkjKmt9C1XYm4KrlWrOXE9rVJvpnEKrPjIXrlKd1hmDGjV',
        #     # offsets_path=f'./{name}/{name}/offsets/{name}.segmentations.offsets.json',
        #     obs_types_from_channel_names=True,
        # )
        # dataset.add_object(segmentations)
        pass
