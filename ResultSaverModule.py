import os
import csv
from datetime import datetime
from LandmarkDatasetModule import VideoLandmarkDataSet

class ResultSaver:
    
    def save_results(self, video_name: str, videoDataSet: VideoLandmarkDataSet) -> None:
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
                    saved = True
        else:
            result_file = "./csv_results/" + base_filename

        self.save_csv(videoDataSet, result_file)

    def save_csv(self, videoDataSet: VideoLandmarkDataSet, file_path: str) -> None:
        dicLandmarks: Dict[int, np.ndarray] = videoDataSet.get_landmarks()
        
        # Determine the maximum number of frames
        max_frames = max(landmarks.shape[0] for landmarks in dicLandmarks.values())

        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write the header
            header = ['Frame'] + [f'L{key}_x' for key in dicLandmarks.keys()] + \
                     [f'L{key}_y' for key in dicLandmarks.keys()] + \
                     [f'L{key}_z' for key in dicLandmarks.keys()] + \
                     [f'L{key}_visible' for key in dicLandmarks.keys()]
            csvwriter.writerow(header)

            # Write the data
            for frame in range(max_frames):
                row = [frame]
                for key in dicLandmarks.keys():
                    if frame < dicLandmarks[key].shape[0]:
                        row.extend(dicLandmarks[key][frame].tolist())
                    else:
                        row.extend(['', '', '', ''])  # Add empty values for missing frames
                csvwriter.writerow(row)
