from abc import ABC, abstractmethod
import cv2
from typing import Any, Dict, Set
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from BodyLandmarkModule import BodyLandmark


class PoseDetectionStrategy(ABC):
    @abstractmethod
    def detect_pose(self, img: cv2.Mat) -> Any:
        pass

    @abstractmethod
    def drawPoseLandmarks(self, img: cv2.Mat, results: Any) -> cv2.Mat:
        pass

    @abstractmethod
    def convertToBodyLandmark(self, results: Any, frame: int) -> Dict[int, BodyLandmark]:
        pass

class MediapipePoseDetectionStrategy(PoseDetectionStrategy):
    def __init__(self, mode: bool = False, uBody: bool = False, smooth: bool = True, 
                 detectionCon: float = 0.5, trackCon: float = 0.5) -> None:
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

        self.target_landmarks: Set[int] = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}

    def detect_pose(self, img: cv2.Mat) -> NormalizedLandmarkList:
        return self.pose.process(img)

    def drawPoseLandmarks(self, img: cv2.Mat, results: NormalizedLandmarkList) -> cv2.Mat:
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(
                img,
                results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS
            )
        return img

    def convertToBodyLandmark(self, results: NormalizedLandmarkList) -> Dict[int, BodyLandmark]:
        landmarks: Dict[int, BodyLandmark] = {}

        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                if idx in self.target_landmarks:
                    landmarks[idx] = BodyLandmark(x=round(lm.x, 3), y=round(lm.y, 3), z=round(lm.z, 3), visible=round(lm.visibility, 3))

        return landmarks


class AnotherPoseDetection(PoseDetectionStrategy):
    def detect_pose(self, img: cv2.Mat) -> NormalizedLandmarkList:
        # Implement another pose detection algorithm
        return NormalizedLandmarkList()  # Return a dummy value for now

    def colorCorrect(self, img: cv2.Mat, conversion: int) -> cv2.Mat:
        return cv2.Mat()
