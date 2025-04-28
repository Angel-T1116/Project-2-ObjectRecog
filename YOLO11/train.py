from ultralytics import YOLO
import torch 

model = YOLO("yolo11x.pt")

# train model using ultralytics
results = model.train(data="coco.yaml", # replace this with whatever dataset file
                        project="runs/detect", # path to place runs in
                        name="train", # folder within the project
                        epochs=100, 
                        imgsz=640,
                        batch=4, # reduce batch size for smaller GPU 
                        device=0) # device=0 for GPU training
