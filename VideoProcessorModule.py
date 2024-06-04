import os
import cv2
import time
from PoseLandmarkDetectorModule import PoseDetector
from LandmarkDatasetModule import VideoLandmarkDataSet

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
        pTime : float
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
    def __init__(self, video_name: str, video_display: bool = True) -> None:
        """
        Initializes the VideoProcessor with the specified video name and display option.

        Parameters
        ----------
        video_name : str
            The name of the video file.
        video_display : bool, optional
            A flag indicating whether to display the video during processing (default is True).
        """
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

    def process_video(self) -> None:
        """
            Processes video frames, detects poses, optionally displays the video, and stores landmark data.
        """
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
