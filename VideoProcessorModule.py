import os
import cv2
import time
from LandmarkDatasetModule import VideoLandmarkDataSet
from PoseDetectionStrategyModule import PoseDetectionStrategy, MediapipePoseDetectionStrategy
from ResultSaverModule import ResultSaver
from BodyLandmarkModule import BodyLandmark
from VideoDisplayerModule import VideoDisplayer

class VideoProcessor:

    def __init__(self, video_name: str, video_display: bool = True, injected_strategy: PoseDetectionStrategy = None) -> None:
        self.video_name: str = video_name
        self.video_display: bool = video_display
        self.video_path: str = self.verify_video_path() # RAISES: FileNotFoundError
        self.video_capture: cv2.Mat = cv2.VideoCapture(self.video_path)

        # Default strategy is MediapipePoseDetectionStrategy if none is provided
        self.strategy: PoseDetectionStrategy = injected_strategy or MediapipePoseDetectionStrategy()
        
        # TODO: For now it stores only one video. Next it has to be able to store a list of videos (process a large dataset of videos)
        self.videoDataSet: VideoLandmarkDataSet = VideoLandmarkDataSet(self.video_path)

        # Initialize the VideoDisplayer
        self.displayer = VideoDisplayer() if self.video_display else None


    def verify_video_path(self) -> str:
        video_path = os.path.join('./videos', self.video_name)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"The video file '{self.video_name}' does not exist in the './videos' directory.")
        else:
            return video_path


    def set_writable_flags(self, img: cv2.Mat, setting: bool) -> None:
        if hasattr(img, "flags"):
            img.flags.writeable = setting


    def process_video(self) -> None:
        success: bool = True
        video_continue: bool = True
        img: cv2.Mat
        success, img = self.video_capture.read()
        frame: int  = 0
        converted: Dict[int, BodyLandmark]

        while success and video_continue:
            self.set_writable_flags(img, False)
            color_corrected = self.colorCorrect(img, cv2.COLOR_BGR2RGB)

            # results variable doesn't have a default dtype
            # Each API should have (or custom implement) a drawing mechanism to draw
            # connections between landmarks. 
            # That method has to work with the same return type from the API.
            results = self.strategy.detect_pose(color_corrected) # AI VISION
            self.set_writable_flags(img, True)
            img = self.strategy.drawPoseLandmarks(img, results)

            converted = self.strategy.convertToBodyLandmark(results)
            self.videoDataSet.addLandmarks(converted)
            frame += 1

            if self.video_display:
                self.displayer.display(img)
                video_continue = self.displayer.check_for_key_press()

            success, img = self.video_capture.read()

        self.video_capture.release()

        if self.video_display:
            self.displayer.close()


    def save_results(self, injected_resultSaver: ResultSaver = None) -> None:
        instance_result_saver: ResultSaver = injected_resultSaver or ResultSaver()
        instance_result_saver.save_results(self.video_name, self.videoDataSet)


    def colorCorrect(self, img: cv2.Mat, conversion: int) -> cv2.Mat:
        return cv2.cvtColor(img, conversion)
