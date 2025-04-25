from ultralytics import YOLO
import torch
import cv2
import os
import pydoc


# Load a model
# model = YOLO("weights\yolo11x.pt")  # load an official model


model = YOLO("runs/detect/train5/weights/best.pt")

results = model.train(data="datasets/lightsaberData.yaml", epochs=100, imgsz=640, batch=4, device=0)

# Predict with the model
# results = model.predict("IMG_4400.mp4", show=True, stream=True, save=True, device='cpu')  # predict on an image

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

""" run video conversion using openCV"""
# videoConverter(inputPath, outputPath)

# generate pydoc documentation file
pydoc.writedoc('main')
