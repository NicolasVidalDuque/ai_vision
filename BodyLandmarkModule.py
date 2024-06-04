import cv2
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark
from typing import List, Dict


class BodyLandmark:
    """
        A class to represent a single landmark.

        Attributes:
            __x (float): The x-coordinate of the landmark.
            __y (float): The y-coordinate of the landmark.
            __z (float): The z-coordinate of the landmark.
            __visible (float): The visibility score of the landmark.
            __frame (int): The frame number the landmark is associated with.
    """

    def __init__(self, mpLandmark: NormalizedLandmark, frame: int) -> None:
        """
            Initializes the Landmark with coordinates and visibility.

            Args:
                mpLandmark (NormalizedLandmark): A MediaPipe NormalizedLandmark object.
                frame (int): The frame number the landmark is associated with.
        """
        self.__x: float = mpLandmark.x
        self.__y: float = mpLandmark.y
        self.__z: float = mpLandmark.z
        self.__visible: float = mpLandmark.visibility
        self.__frame: int = frame

    # Getter methods
    def get_x(self) -> float:
        return self.__x

    def get_y(self) -> float:
        return self.__y

    def get_z(self) -> float:
        return self.__z

    def get_visible(self) -> float:
        return self.__visible

    def get_frame(self) -> int:
        return self.__frame

    def __str__(self) -> str:
        """
        Returns a string representation of the landmark.
        """
        return f"(x={self.__x}, y={self.__y}, z={self.__z}, visible={self.__visible}, frame={self.__frame})"