from VideoProcessorModule import *


def main() -> None:
    """
        The main function to create instances of VideoProcessor and ResultSaver, process the video, and save the results.
    """
    video_name = 'yoga1.mp4'
    video_processor = VideoProcessor(video_name=video_name)
    video_processor.process_video()
    video_processor.save_results()


if __name__ == "__main__":
    main()
