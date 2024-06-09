from typing import Dict, Tuple

class BodyLandmark:
    def __init__(self) -> None:
        self.__x: float = 0.0
        self.__y: float = 0.0
        self.__z: float = 0.0
        self.__visible: float = 0.0
        self.__frame: int = 0

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

    def set_x(self, x: float) -> None:
        self.__x = x

    def set_y(self, y: float) -> None:
        self.__y = y

    def set_z(self, z: float) -> None:
        self.__z = z

    def set_visible(self, visible: float) -> None:
        self.__visible = visible

    def set_frame(self, frame: int) -> None:
        self.__frame = frame

    def set_all(self, x: float, y: float, z: float, visible: float, frame: int) -> None:
        self.__x = x
        self.__y = y
        self.__z = z
        self.__visible = visible
        self.__frame = frame

    def get_all(self) -> tuple[float, float, float, float, int]:
        return self.__x, self.__y, self.__z, self.__visible, self.__frame

    def __str__(self) -> str:
        return f"(x={self.__x}, y={self.__y}, z={self.__z}, visible={self.__visible}, frame={self.__frame})"