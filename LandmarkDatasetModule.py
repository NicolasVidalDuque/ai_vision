from typing import List, Dict
from BodyLandmarkModule import BodyLandmark

class VideoLandmarkDataSet:
    """
        This data structure has:
            key: int: each body landmark index (shoulder -> 1, head -> 33)... see API documentation to see each landmark id
            List[BodyLandmark]: one column corresponds to one frame. Each frame has all landmarks positions and attributes at that specific frame.

            Therefore one value in the "matrix" corresponds to (row, col) : (frame, landmark_id)

        TODO: 
            Create new method: to_csv()
    """

    def __init__(self, file_name: str) -> None:
        self.dicLandmarks: Dict[int, List[BodyLandmark]] = {}
        self.file_name: str = file_name

    def addLandmarks(self, landmarkDict: Dict[int, BodyLandmark], frame: int) -> None:
        for key, value_lm in landmarkDict.items():
            if key not in self.dicLandmarks:
                self.dicLandmarks[key] = []
            self.dicLandmarks[key].append(value_lm)

    def get_landmarks(self) -> Dict[int, List[BodyLandmark]]:
        return self.dicLandmarks
