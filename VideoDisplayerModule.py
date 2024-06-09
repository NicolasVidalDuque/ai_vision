import cv2
import time

class VideoDisplayer:
    def __init__(self) -> None:
        self.previous_time: float = time.time()

    def calculate_fps(self, previous_time: float, current_time: float) -> float:
        fps = 1 / (current_time - previous_time)
        return fps

    def write_fps(self, img: cv2.Mat, fps: float) -> None:
        cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    def display(self, img: cv2.Mat) -> None:
        current_time = time.time()
        fps = self.calculate_fps(self.previous_time, current_time)
        self.previous_time = current_time
        self.write_fps(img, fps)
        cv2.imshow("Image", img)
        
    def check_for_key_press(self) -> bool:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    def close(self) -> None:
        cv2.destroyAllWindows()
