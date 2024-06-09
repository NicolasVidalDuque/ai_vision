import cv2
import mediapipe as mp
import time
from typing import Tuple, Dict, Any
import pandas as pd
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from LandmarkDatasetModule import VideoLandmarkDataSet
from PoseDetectionStrategyModule import PoseDetectionStrategy, MediapipePoseDetectionStrategy

class PoseDetector:

    def __init__(self, strategy: PoseDetectionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: PoseDetectionStrategy):
        self.strategy = strategy








    def colorCorrect(self, img: cv2.Mat, conversion: int) -> cv2.Mat:
        """
            Converts the color space of an image.

            Args:
                img (cv2.Mat): The input image.
                conversion (int): The OpenCV color conversion code.

            Returns:
                cv2.Mat: The color-corrected image.
        """
        return cv2.cvtColor(img, conversion)

