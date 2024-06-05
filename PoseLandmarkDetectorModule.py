import cv2
import mediapipe as mp
import time
from typing import Tuple, Dict, Any
import pandas as pd
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from LandmarkDatasetModule import VideoLandmarkDataSet
from PoseDetectionStrategyModule import PoseDetectionStrategy, MediapipePoseDetectionStrategy

class PoseDetector:
    """
        A class to detect human poses using the MediaPipe library.
        
        Attributes:
            mode (bool): Whether to treat input images as a batch of static images.
            uBody (bool): Whether to enable segmentation for upper body only.
            smooth (bool): Whether to smooth landmarks over time.
            detectionCon (float): Minimum confidence value for person detection to be considered successful.
            trackCon (float): Minimum confidence value for the pose landmarks to be considered tracked successfully.
    """

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

