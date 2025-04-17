from ultralytics import YOLO
import torch
import cv2
import os

def videoConverter(inputPath, outputPath):
    """Converts video in .avi format to .mp4"""
    
    vid = cv2.VideoCapture(inputPath)

    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_size = (width, height)

    output = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, vid_size)

    while(True):
        ret, frame = vid.read()
        if not (ret):
            break
        output.write(frame)

    vid.release()
    output.release()
        
# Load a model
model = YOLO("weights\yolo11x.pt")  # load an official model


# Predict with the model
results = model.predict("IMG_4400.mp4", show=True, stream=True, save=True, device='cpu')  # predict on an image

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box

inputPath = r"runs\detect\predict\IMG_4400.avi"
outputPath = r"runs\detect\predict\results.mp4"

videoConverter(inputPath, outputPath)