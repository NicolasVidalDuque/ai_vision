import os
import cv2
import time
from PoseLandmarkDetectorModule import PoseDetector
from LandmarkDatasetModule import VideoLandmarkDataSet
from ResultSaverModule import ResultSaver

class VideoProcessor:

    def __init__(self, video_name: str, video_display: bool = True) -> None:

        self.video_name = video_name
        self.video_display = video_display
        self.verify_video_path()
        self.video_path = os.path.join('./videos', self.video_name)
        self.cap = cv2.VideoCapture(self.video_path)
        self.detector = PoseDetector()
        self.pTime = time.time()
        self.frame = 0
        self.videoDataSet = VideoLandmarkDataSet(self.video_path)

    def verify_video_path(self) -> None:

        video_path = os.path.join('./videos', self.video_name)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"The video file '{self.video_name}' does not exist in the './videos' directory.")
    
    def set_video_display(self, video_display: bool) -> None:

        self.video_display = video_display

    def process_video(self) -> None:

        success, img = self.cap.read()

        while success:
            img.flags.writeable  = False
            color_corrected = self.detector.colorCorrect(img, cv2.COLOR_BGR2RGB)
            results = self.detector.findPoseLandmarks(color_corrected)
            img.flags.writeable  = True
            img = self.detector.drawPoseLandmarks(img, results)
            
            if self.video_display:
                cTime = time.time()
                fps, self.pTime = self.detector.calculateFPS(self.pTime, cTime)
                self.write_fps(img, fps)
                cv2.imshow("Image", img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            success, img = self.cap.read()
            self.videoDataSet.addLandmarks(results.pose_landmarks, self.frame)
            self.frame += 1

        self.cap.release()
        cv2.destroyAllWindows()

    def write_fps(self, img: cv2.Mat, fps: float) -> None:
        cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    def save_results(self, inyection_result_saver: ResultSaver = None) -> None:
        instance_result_saver: ResultSaver = inyection_result_saver or ResultSaver()
        instance_result_saver.save_results(self.video_name, self.videoDataSet)
