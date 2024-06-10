from typing import Dict
from BodyLandmarkModule import BodyLandmark
import numpy as np

class VideoLandmarkDataSet:
    def __init__(self, file_name: str) -> None:
        self.dicLandmarks: Dict[int, np.ndarray] = {}
        self.file_name: str = file_name


    def addLandmarks(self, landmarkDict: Dict[int, BodyLandmark]) -> None:
        for key, value_lm in landmarkDict.items():
            if key not in self.dicLandmarks:
                self.dicLandmarks[key] = np.empty((0, 4))
            new_landmark = np.array([[value_lm.x, value_lm.y, value_lm.z, value_lm.visible]])
            self.dicLandmarks[key] = np.append(self.dicLandmarks[key], new_landmark, axis=0)


    def get_landmarks(self) -> Dict[int, np.ndarray]:
        return self.dicLandmarks
