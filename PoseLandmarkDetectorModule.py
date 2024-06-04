import cv2
import mediapipe as mp
import time
from typing import Tuple, Dict, Any
import pandas as pd
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from LandmarkDatasetModule import VideoLandmarkDataSet

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

    def __init__(self, mode: bool = False, uBody: bool = False, smooth: bool = True, 
                 detectionCon: float = 0.5, trackCon: float = 0.5) -> None:
        """
            Initializes the PoseDetector with the given configuration.

            Args:
                mode (bool): Static image mode.
                uBody (bool): Enable upper body segmentation.
                smooth (bool): Smooth landmarks.
                detectionCon (float): Minimum detection confidence.
                trackCon (float): Minimum tracking confidence.
        """
        self.mode = mode
        self.uBody = uBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=self.uBody,
            smooth_segmentation=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPoseLandmarks(self, img: cv2.Mat) -> NormalizedLandmarkList:
        """
            Finds pose landmarks in an image. This is the AI part

            Args:
                img (cv2.Mat): The input image.

            Returns:
                results: The results object containing pose landmarks.


        """
        results = self.pose.process(img)
        return results

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

    def drawPoseLandmarks(self, img: cv2.Mat, results: NormalizedLandmarkList) -> cv2.Mat:
        """
            Draws pose landmarks on an image.

            Args:
                img (cv2.Mat): The input image.
                results: The results object containing pose landmarks.

            Returns:
                cv2.Mat: The image with pose landmarks drawn.
        """
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(
                img,
                results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS
            )
        return img

    def calculateFPS(self, pTime: float, cTime: float) -> Tuple[float, float]:
        """
            Calculates frames per second (FPS).

            Args:
                pTime (float): Previous timestamp.
                cTime (float): Current timestamp.

            Returns:
                Tuple[float, float]: The calculated FPS and the current timestamp.
        """
        fps = 1 / (cTime - pTime)
        return fps, cTime

