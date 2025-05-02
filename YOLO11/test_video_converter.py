import cv2
import os
import numpy as np
from YOLO11.predict import videoConverter
import pydoc



""" unit test for video converter"""
def test_videoConverter():
    """ Create video for testing"""
    os.makedirs("runs/test", exist_ok=True)

    test_path = "runs/test/test.avi"
    output_path = "runs/test/test.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(test_path, fourcc, 10.0, (120, 120))

    for _ in range(20):
        frame = 255 * np.ones((120, 120, 3), dtype=np.uint8)
        out.write(frame)


    out.release()
    videoConverter(test_path, output_path)
    assert os.path.exists(output_path), "Video converter did not create an output video."

pydoc.writedoc('test_video_converter')