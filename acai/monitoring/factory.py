from enum import Enum

from ..data import generate_batch
from .interpolation import InterpolationMonitoring
from .reconstruction import ReconstructionMonitoring


class MonitoringType:
    interpolation = InterpolationMonitoring
    reconstruction = ReconstructionMonitoring


class MonitoringFactory:
    def build(self, monitoring_type: MonitoringType, n_images: int):
        if monitoring_type == MonitoringType.interpolation:
            images = generate_batch(2)
            return InterpolationMonitoring(images[0], images[1])
        elif monitoring_type == MonitoringType.reconstruction:
            images = generate_batch(n_images)
            return ReconstructionMonitoring(images)
        else:
            raise NotImplementedError()
