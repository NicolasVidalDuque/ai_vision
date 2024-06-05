from abc import ABC, abstractmethod
import cv2
from typing import Any
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

class PoseDetectionStrategy(ABC):
    """
        TODO: The return type currently is Any but it should be some kind of NormalizedLandmarkList
              That is exclusive to MediaPipe. When the ai API gets changed the return DataType will change.
    """
    @abstractmethod
    def detect_pose(self, img: cv2.Mat) -> Any:
        pass

    @abstractmethod
    def drawPoseLandmarks(self, img: cv2.Mat, results: NormalizedLandmarkList) -> cv2.Mat:
        pass


class MediapipePoseDetectionStrategy(PoseDetectionStrategy):
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

    def detect_pose(self, img: cv2.Mat) -> NormalizedLandmarkList:
        results = self.pose.process(img)
        return results

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

class AnotherPoseDetection(PoseDetectionStrategy):
    def detect_pose(self, img: cv2.Mat) -> NormalizedLandmarkList:
        # Implement another pose detection algorithm
        return NormalizedLandmarkList()  # Return a dummy value for now

    def colorCorrect(self, img: cv2.Mat, conversion: int) -> cv2.Mat:
        return cv2.Mat()
