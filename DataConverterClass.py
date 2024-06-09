from typing import Any, Dict
from PoseDetectionStrategyModule import PoseDetectionStrategy
from BodyLandmarkModule import BodyLandmark

class DataConverter:
    def __init__(self, strategy: PoseDetectionStrategy) -> None:
        self.strategy = strategy

    def convert(self, results: Any, frame: int) -> Dict[int, BodyLandmark]:
        return self.strategy.convertToBodyLandmark(results, frame)
