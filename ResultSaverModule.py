import os
import pandas as pd
from datetime import datetime
from LandmarkDatasetModule import VideoLandmarkDataSet

class ResultSaver:
    """
        A class used to save video analysis results to a CSV file.

        Attributes
        ----------
        video_name : str
            The name of the video file.
        videoDataSet : VideoLandmarkDataSet
            The dataset for storing landmarks.

        Methods
        -------
        save_results():
            Saves the landmark data to a CSV file with the video name and timestamp.
    """
    def __init__(self, video_name: str, videoDataSet: VideoLandmarkDataSet) -> None:
        """
            Initializes the ResultSaver with the specified video name and dataset.

            Parameters
            ----------
            video_name : str
                The name of the video file.
            videoDataSet : VideoLandmarkDataSet
                The dataset for storing landmarks.
        """
        self.video_name = video_name
        self.videoDataSet = videoDataSet

    def save_results(self) -> None:
        """
            Saves the landmark data to a CSV file with the video name and timestamp.
        """
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
