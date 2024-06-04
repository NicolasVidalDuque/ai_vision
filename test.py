from VideoProcessorModule import *
from ResultSaverModule import *

def main() -> None:
    """
        The main function to create instances of VideoProcessor and ResultSaver, process the video, and save the results.
    """
    video_name = 'yoga1.mp4'
    video_processor = VideoProcessor(video_name=video_name)
    video_processor.process_video()

    result_saver = ResultSaver(video_name=video_name, videoDataSet=video_processor.videoDataSet)
    result_saver.save_results()

if __name__ == "__main__":
    main()
