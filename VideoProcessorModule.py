import os
import cv2
import time
from PoseLandmarkDetectorModule import PoseDetector
from LandmarkDatasetModule import VideoLandmarkDataSet
from PoseDetectionStrategyModule import PoseDetectionStrategy, MediapipePoseDetectionStrategy
from ResultSaverModule import ResultSaver

class VideoProcessor:
    """
        A class used to process video frames for pose detection.

        Attributes
        ----------
        video_name : str
            The name of the video file.
        video_display : bool
            A flag indicating whether to display the video during processing.
        video_path : str
            The full path to the video file.
        cap : cv2.VideoCapture
            The video capture object.
        detector : PoseDetector
            The pose detector object.
        previousTime : float
            The previous timestamp for FPS calculation.
        frame : int
            The current frame count.
        videoDataSet : VideoLandmarkDataSet
            The dataset for storing landmarks.

        Methods
        -------
        _verify_video_path():
            Verifies if the video file exists in the './videos' directory.
        set_video_display(video_display: bool):
            Sets the video display flag.
        process_video():
            Processes video frames, detects poses, optionally displays the video, and stores landmark data.
        _write_fps(img: cv2.Mat, fps: float):
            Writes the FPS value on the image.
    """
    def __init__(self, video_name: str, video_display: bool = True, strategy: PoseDetectionStrategy = None) -> None:
        """
        Initializes the VideoProcessor with the specified video name and display option.

        Parameters
        ----------
        video_name : str
            The name of the video file.
        video_display : bool, optional
            A flag indicating whether to display the video during processing (default is True).
        """
        self.video_name: str = video_name
        self.video_display: bool = video_display
        self._verify_video_path()
        self.video_path: str = os.path.join('./videos', self.video_name)
        self.video_capture: cv2.Mat = cv2.VideoCapture(self.video_path)

        # Default strategy is MediapipePoseDetectionStrategy if none is provided
        if strategy is None:
            strategy = MediapipePoseDetectionStrategy()
        self.detector: PoseDetector = PoseDetector(strategy)

        self.previousTime: float = time.time()
        self.frame: int  = 0
        
        # TODO: For now it stores only one video. Next it has to be able to store a list of videos (process a large dataset of videos)
        self.videoDataSet: VideoLandmarkDataSet = VideoLandmarkDataSet(self.video_path)

    def _verify_video_path(self) -> None:
        """
            Verifies if the video file exists in the './videos' directory.
            
            Raises
            ------
            FileNotFoundError
                If the video file does not exist.
        """
        video_path = os.path.join('./videos', self.video_name)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"The video file '{self.video_name}' does not exist in the './videos' directory.")
    
    def set_video_display(self, video_display: bool) -> None:
        """
            Sets the video display flag.

            Parameters
            ----------
            video_display : bool
                A flag indicating whether to display the video during processing.
        """
        self.video_display = video_display

    def set_writable_flags(self, img: cv2.Mat, setting: bool) -> None:
        if hasattr(img, "flags"):
            img.flags.writeable = setting

    def process_video(self) -> None:
        """
            Processes video frames, detects poses, optionally displays the video, and stores landmark data.
        """
        success: bool = True
        img: cv2.Mat
        success, img = self.video_capture.read()

        while success:

            self.set_writable_flags(img, False)
            # TODO: Create a method to identify which color coding does the original video has.
            #       From that, determine if a conversion is needed
            color_corrected = self.detector.colorCorrect(img, cv2.COLOR_BGR2RGB)
            results = self.detector.strategy.detect_pose(color_corrected) # AI VISION
            self.set_writable_flags(img, True)
            img = self.detector.strategy.drawPoseLandmarks(img, results)
            
            self.videoDataSet.addLandmarks(results.pose_landmarks, self.frame)
            self.frame += 1

            # TODO: Create a new class VideoDisplayer which does everything for video display.
            if self.video_display:
                currentTime = time.time()
                fps = self.calculateFPS(self.previousTime, currentTime)
                self.previousTime = currentTime
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
        """
            Writes the FPS value on the image.

            Parameters
            ----------
            img : cv2.Mat
                The image on which to write the FPS value.
            fps : float
                The frames per second value to write.
        """
        cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    def save_results(self, inyection_result_saver: ResultSaver = None) -> None:
        instance_result_saver: ResultSaver = inyection_result_saver or ResultSaver()
        instance_result_saver.save_results(self.video_name, self.videoDataSet)