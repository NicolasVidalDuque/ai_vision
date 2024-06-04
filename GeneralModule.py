import os
import cv2
import time
import pandas as pd
import random as rd
from datetime import datetime
from typing import List, Dict, Tuple, Any

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark

class PoseDetector:
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

    def findPoseLandmarks(self, img: cv2.Mat) -> Any:
        results = self.pose.process(img)
        return results

    def colorCorrect(self, img: cv2.Mat, conversion: int) -> cv2.Mat:
        return cv2.cvtColor(img, conversion)

    def drawPoseLandmarks(self, img: cv2.Mat, results: NormalizedLandmarkList) -> cv2.Mat:
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(
                img,
                results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS
            )
        return img

    def calculateFPS(self, pTime: float, cTime: float) -> Tuple[float, float]:
        fps = 1 / (cTime - pTime)
        return fps, cTime

class VideoLandmarkDataSet:
    def __init__(self, file_name: str) -> None:
        self.dicLandmarks: Dict[str, List[BodyLandmark]] = {}
        self.file_name: str = file_name

    def addLandmarks(self, mpLandmarks: NormalizedLandmarkList, frame: int) -> None:
        for num, lm in enumerate(mpLandmarks.landmark):
            key = str(num)
            if key not in self.dicLandmarks:
                self.dicLandmarks[key] = []
            self.dicLandmarks[key].append(BodyLandmark(lm, frame))

    def toPandas(self) -> pd.DataFrame:
        data = {}
        
        for key, landmarks in self.dicLandmarks.items():
            frame_data = []
            for landmark in landmarks:
                frame_data.append(str(landmark))
            
            data[key] = frame_data
        
        df = pd.DataFrame(data)
        
        return df   



class BodyLandmark:
    def __init__(self, mpLandmark: NormalizedLandmark, frame: int) -> None:
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
        return f"(x={self.__x}, y={self.__y}, z={self.__z}, visible={self.__visible}, frame={self.__frame})"


class _VideoAnalyzer:
    def __init__(self, video_name: str, video_display: bool = True) -> None:
        self.video_name = video_name
        self.video_display = video_display
        self._verify_video_path()
        self.video_path = os.path.join('./videos', self.video_name)
        self.cap = cv2.VideoCapture(self.video_path)
        self.detector = PoseDetector()
        self.pTime = time.time()
        self.frame = 0
        self.videoDataSet = VideoLandmarkDataSet(self.video_path)

    def _verify_video_path(self) -> None:
        video_path = os.path.join('./videos', self.video_name)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"The video file '{self.video_name}' does not exist in the './videos' directory.")
    
    def set_video_display(self, video_display: bool) -> None:
        self.video_display = video_display

    def analyze_video(self) -> None:
        success, img = self.cap.read()

        while success:
            color_corrected = self.detector.colorCorrect(img, cv2.COLOR_BGR2RGB)
            results = self.detector.findPoseLandmarks(color_corrected)
            img = self.detector.drawPoseLandmarks(img, results)
            
            if self.video_display:
                cTime = time.time()
                fps, self.pTime = self.detector.calculateFPS(self.pTime, cTime)
                self._write_fps(img, fps)
                cv2.imshow("Image", img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            success, img = self.cap.read()
            self.videoDataSet.addLandmarks(results.pose_landmarks, self.frame)
            self.frame += 1

        self.cap.release()
        cv2.destroyAllWindows()

    def _write_fps(self, img: cv2.Mat, fps: float) -> None:
        cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    def save_results(self) -> None:
        df: pd.DataFrame = self.videoDataSet.toPandas()
        date_str: str = datetime.now().strftime("%d-%H-%M")
        base_filename: str = f"results_{self.video_name.split('.')[0]}_{date_str}.csv"
        
        # Check if the file already exists
        if os.path.exists("./csv_results/" + base_filename):
            version: int = 0
            saved: bool = False
            while not saved:
                version += 1
                new_filename = f"results_{self.video_name.split('.')[0]}_{date_str}_{version}.csv"
                if not os.path.exists("./csv_results/" + new_filename):
                    result_file = "./csv_results/" + new_filename
                    saved = not saved
        else:
            result_file = "./csv_results/" + base_filename

        df.to_csv(result_file)

def main() -> None:
    video_name = 'yoga1.mp4'
    analyzer = _VideoAnalyzer(video_name=video_name)
    analyzer.analyze_video()
    analyzer.save_results()

if __name__ == "__main__":
    main()
