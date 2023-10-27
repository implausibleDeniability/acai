from enum import Enum

from ..data.dataloader import DataLoaderBase
from .interpolation import InterpolationMonitoring
from .reconstruction import ReconstructionMonitoring


class MonitoringType:
    interpolation = InterpolationMonitoring
    reconstruction = ReconstructionMonitoring


class MonitoringFactory:
    def build(self, monitoring_type: MonitoringType, dataloader: DataLoaderBase, n_images: int):
        if monitoring_type == MonitoringType.interpolation:
            images = dataloader.get_eval_batch(n_images * 2)
            return InterpolationMonitoring(images[:n_images], images[n_images:])
        elif monitoring_type == MonitoringType.reconstruction:
            images = dataloader.get_eval_batch(n_images * 2)
            return ReconstructionMonitoring(images)
        else:
            raise NotImplementedError()
