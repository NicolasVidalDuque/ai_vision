# Body Landmark Trajectory Analysis


![Demo](https://www.youtube.com/watch?v=UAN1R-tQccI)

https://www.youtube.com/watch?v=UAN1R-tQccI

This project is designed to analyze exercise form using video input. It extracts body landmarks from MP4 video files, saves them as CSV files, and provides visualizations of the body landmark trajectories over time. The goal is to develop a system that can compare a user's form to that of an expert, providing feedback on how to improve the exercise technique.

## Project Structure

The project files and directories are organized as follows:

```plaintext
AI_VISION
│   BodyLandmarkModule.py            # Module for handling body landmark extraction logic
│   body_landmark_demo.mp4           # Demo video for testing the body landmark extraction
│   deleteVideoResults.py            # Script to delete old video results
│   LandmarkDatasetModule.py         # Module for managing landmark datasets
│   main.py                          # Main script to run the exercise analysis process
│   PoseDetectionStrategyModule.py   # Module defining various pose detection strategies
│   README.md                        # Project documentation (this file)
│   ResultSaverModule.py             # Module to save results (e.g., landmarks) to files
│   VideoDisplayerModule.py          # Module for displaying video with detected landmarks
│   VideoProcessorModule.py          # Module for processing video input frame by frame
│
├───csv_results                      # Directory for storing CSV files with landmark data
│       results_yoga1_09-21-05.csv
│       results_yoga1_10-09-00.csv
│       results_yoga1_29-14-29.csv
│       results_yoga1_29-14-38.csv
│
├───trajectory_analisys              # Directory for trajectory analysis files
│       analisys_1.ipynb             # Jupyter notebook for analyzing landmark trajectories
│       yoga_results.gif             # GIF visualization of landmark trajectory
│
└───videos                           # Directory for storing input videos for analysis
        yoga1.mp4
```


## Modules Overview

- **BodyLandmarkModule.py:** Contains the logic for extracting body landmarks from video frames.
- **LandmarkDatasetModule.py:** Handles the creation and management of datasets from extracted landmark data.
- **PoseDetectionStrategyModule.py:** Implements different strategies for pose detection.
- **ResultSaverModule.py:** Handles saving the results, such as landmark data, into CSV files.
- **VideoProcessorModule.py:** Processes video input and applies landmark extraction algorithms.
- **VideoDisplayerModule.py:** Provides functionality for displaying the processed video with overlaid landmarks.

## Data Analysis

- **Trajectory Analysis:** After the body landmarks are extracted and saved as CSV files, the `trajectory_analisys` directory contains tools like Jupyter notebooks (`analisys_1.ipynb`) to visualize and analyze the movement of specific body parts over time.

## Future Work

- **Comparison to Expert Models:** Upcoming features will include a database of expert form data to compare user movements against predefined "perfect" trajectories.
- **Feedback System:** Develop a feedback mechanism that suggests improvements to the user's exercise form based on the comparison to expert data.
