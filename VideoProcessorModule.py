import os
import cv2
import time
from LandmarkDatasetModule import VideoLandmarkDataSet
from PoseDetectionStrategyModule import PoseDetectionStrategy, MediapipePoseDetectionStrategy
from ResultSaverModule import ResultSaver
from BodyLandmarkModule import BodyLandmark

class VideoProcessor:

    def __init__(self, video_name: str, video_display: bool = True, injectected_strategy: PoseDetectionStrategy = None) -> None:

        self.video_name: str = video_name
        self.video_display: bool = video_display
        self.video_path: str = self.verify_video_path() # RAISES: FileNotFoundError
        self.video_capture: cv2.Mat = cv2.VideoCapture(self.video_path)

        # Default strategy is MediapipePoseDetectionStrategy if none is provided
        self.strategy: PoseDetectionStrategy = injectected_strategy or MediapipePoseDetectionStrategy()
        
        # TODO: For now it stores only one video. Next it has to be able to store a list of videos (process a large dataset of videos)
        self.videoDataSet: VideoLandmarkDataSet = VideoLandmarkDataSet(self.video_path)


    def verify_video_path(self) -> str:
        video_path = os.path.join('./videos', self.video_name)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"The video file '{self.video_name}' does not exist in the './videos' directory.")
        else:
            return video_path
    

    def set_video_display(self, video_display: bool) -> None:
        self.video_display = video_display


    def set_writable_flags(self, img: cv2.Mat, setting: bool) -> None:
        if hasattr(img, "flags"):
            img.flags.writeable = setting


    def process_video(self) -> None:

        success: bool = True
        img: cv2.Mat
        success, img = self.video_capture.read()
        previous_time: float = time.time()

        frame: int  = 0
        converted: Dict[int, BodyLandmark]

        while success:

            self.set_writable_flags(img, False)
            # TODO: Create a method to identify which color coding does the original video has.
            #       From that, determine if a conversion is needed
            color_corrected = self.colorCorrect(img, cv2.COLOR_BGR2RGB)

            """
                results variable doesn't have a default dtype

                Each API should have (or custom implement) a drawing mechanism to draw
                connections between landmarks. 

                That method has to work with the same return type from the API.
            """
            results = self.strategy.detect_pose(color_corrected) # AI VISION
            self.set_writable_flags(img, True)
            img = self.strategy.drawPoseLandmarks(img, results)

            converted = self.strategy.convertToBodyLandmark(results, frame)
            
            self.videoDataSet.addLandmarks(converted, frame)
            frame += 1

            # TODO: Create a new class VideoDisplayer which does everything for video display.
            if self.video_display:
                currentTime = time.time()
                fps = self.calculateFPS(previous_time, currentTime)
                previous_time = currentTime
                self.write_fps(img, fps)
                cv2.imshow("Image", img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            success, img = self.video_capture.read()

        self.video_capture.release()
        cv2.destroyAllWindows()

    def calculateFPS(self, previousTime: float, currentTime: float) -> float:
        fps = 1 / (currentTime - previousTime)
        return fps

    def write_fps(self, img: cv2.Mat, fps: float) -> None:

        cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    def save_results(self, injected_resultSaver: ResultSaver = None) -> None:
        instance_result_saver: ResultSaver = injected_resultSaver or ResultSaver()
        instance_result_saver.save_results(self.video_name, self.videoDataSet)

    def colorCorrect(self, img: cv2.Mat, conversion: int) -> cv2.Mat:

        return cv2.cvtColor(img, conversion)