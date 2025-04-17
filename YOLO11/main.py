from ultralytics import YOLO
import torch

# Load a model
model = YOLO("weights\yolo11x.pt")  # load an official model


# Predict with the model
results = model.predict("IMG_4400.mp4", show=True, stream=True, save=True, device='cpu')  # predict on an image
print(results)

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box