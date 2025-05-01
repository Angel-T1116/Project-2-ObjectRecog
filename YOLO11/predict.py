from ultralytics import YOLO
import cv2
import os
import numpy as np
import shutil
import pydoc
import webbrowser

#TO DO: take output path from results and use as input for videoConverter


def clean_runs(folder):
    """ This function cleans up old runs from prediction """
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted '{folder} directory")
    else: 
        print(f"{folder} not found")


def videoConverter(inputPath, outputPath):
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


if __name__ == "__main__": 

    folder = "runs/detect/predict"
    clean_runs(folder)
    # Load a model
    model = YOLO("weights/yolo11x.pt")  # load a trained model


    # Predict with the model for video input
    inputVid = "IMG_4400"
    inputExt = ".mp4"

    # predict for images in directory
    results = model.predict(source=f"datasets/{inputVid}{inputExt}", project="runs/detect", name="predict", stream=True, show=True, save=True, device=0)  # predict on an image


    # Access the results
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box

    inputPath = f"runs/detect/predict/{inputVid}.avi" # edit this as necessary to the expected path of the output video
    outputPath = f"runs/detect/predict/{inputVid}.mp4"  

    """ run video conversion using openCV"""
    videoConverter(inputPath, outputPath)

    pydoc.writedoc(predict)
    
