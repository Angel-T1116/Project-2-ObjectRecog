from ultralytics import YOLO
import torch 

""" File containing training parameters used to generate model"""
model = YOLO("yolo11x.pt") # load pretrained model

# train model using ultralytics
results = model.train(data="data.yaml", # ensure you have the dataset downloaded before using this
                        project="runs/detect", # path to place runs in
                        name="train", # folder within the project
                        epochs=100, 
                        imgsz=640,
                        batch=4, # reduce batch size for smaller GPU 
                        device=0) # device=0 for GPU training
