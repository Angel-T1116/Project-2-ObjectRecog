from ultralytics import YOLO
import torch 

model = YOLO("yolo11x.pt")


results = model.train(data="datasets/lightsaberData.yaml", epochs=100, imgsz=640, batch=4, device=0)


