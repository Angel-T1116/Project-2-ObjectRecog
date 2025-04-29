from ultralytics import YOLO
import cv2
import os
import numpy as np

import pydoc
# TO DO: clear previous predict runs
#TO DO: take output path from results and use as input for videoConverter

"""
This function converts the video output from ultralytics in a .avi format to .mp4
"""
def CustomVideoConverter(inputPath, outputPath):
    """Converts video in .avi format to .mp4"""
    
    # convert the video into a cv2 input
    vid = cv2.VideoCapture(inputPath)

    # get data from the video
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_size = (width, height)

    # create an output file in mp4 format
    output = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, vid_size)

    # write the frames into the output video
    while(True):
        ret, frame = vid.read()
        if not (ret):
            break
        output.write(frame)

    vid.release()
    output.release()

# pydoc.writedoc('videoConverter')
        
# Load a model
model = YOLO("weights/best.pt")  # load a trained model


# Predict with the model for video input
# results = model.predict("IMG_4400.mp4", show=True, stream=True, save=True, device=0)  # predict on an image

# predict for images in directory
results = model.predict(source="datasets/IMG_4400.mp4", project="runs/detect", name="predict", stream=True, show=True, save=True, device=0)  # predict on an image


# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box

inputPath = "YOLO11/runs/detect/predict/IMG_4400.avi" # edit this as necessary to the expected path of the output video
outputPath ="YOLO11/runs/detect/predict/results.mp4"  

""" run video conversion using openCV"""
CustomVideoConverter(inputPath, outputPath)

# generate pydoc documentation file
# pydoc.writedoc('predict')
