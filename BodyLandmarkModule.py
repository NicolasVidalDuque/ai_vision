from dataclasses import dataclass, field

@dataclass
class BodyLandmark:
    x: float
    y: float
    z: float
    visible: float

    def get_all(self) -> tuple:
        return (self.x, self.y, self.z, self.visible)

    def __repr__(self) -> str:
        return f"BodyLandmark(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, visible={self.visible:.3f})"