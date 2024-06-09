import cv2
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark
from typing import List, Dict
import pandas as pd

from BodyLandmarkModule import BodyLandmark

class VideoLandmarkDataSet:
    """

        This data structure has:
            key: str: each body landmark index (shoulder -> 1, head -> 33)... see API documentation to see each landmark id
            List[BodyLandmark]]: one column corresponds to one frame. Each frame has all landmarks positions and attributes at that specific frame.

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

    def toPandas(self) -> pd.DataFrame:

        # Create a dictionary to hold the data for the DataFrame
        data = {}
        
        # Iterate over each landmark number (key)
        for key, landmarks in self.dicLandmarks.items():
            # Initialize an empty list to store string representations of landmarks for each frame
            frame_data = []
            for landmark in landmarks:
                frame_data.append(str(landmark))
            
            # Add the list to the data dictionary with the key as the row identifier
            data[key] = frame_data
        
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(data)
        
        return df   
