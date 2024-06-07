import os
import pandas as pd
from datetime import datetime
from LandmarkDatasetModule import VideoLandmarkDataSet

class ResultSaver:

    def save_results(self, video_name: str, videoDataSet: VideoLandmarkDataSet) -> None:

        df: pd.DataFrame = videoDataSet.toPandas()
        date_str: str = datetime.now().strftime("%d-%H-%M")
        base_filename: str = f"results_{video_name.split('.')[0]}_{date_str}.csv"
        
        # Check if the file already exists
        if os.path.exists("./csv_results/" + base_filename):
            version: int = 0
            saved: bool = False
            while not saved:
                version += 1
                new_filename = f"results_{video_name.split('.')[0]}_{date_str}_{version}.csv"
                if not os.path.exists("./csv_results/" + new_filename):
                    result_file = "./csv_results/" + new_filename
                    saved = not saved
        else:
            result_file = "./csv_results/" + base_filename

        df.to_csv(result_file)