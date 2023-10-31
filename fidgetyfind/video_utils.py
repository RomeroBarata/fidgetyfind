import cv2 as cv


def get_video_fps(video_filepath):
    """Given a video filepath, return the video's FPS."""
    cap = cv.VideoCapture(video_filepath)
    frame_rate = cap.get(cv.CAP_PROP_FPS)
    cap.release()
    return frame_rate
