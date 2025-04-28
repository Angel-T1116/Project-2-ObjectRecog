from ultralytics import YOLO


""" This is an example function to use transfer learning with the ultralytics YOLO11 model. """

# model = YOLO("yolo11n.yaml") # build a new model from yaml
# model = YOLO("yolo11n.pt") # load pretrained model
model = YOLO("yolo11x.pt") # build from yaml and transfer weights

results = model.train(data="datasets/data.yaml",
                        project="runs/detect",
                        name="train",
                        epochs=100,
                        imgsz=640,  
                        workers=2, #reduce number of workers for laptop processing
                        batch=4,
                        freeze=[0, -2], # freeze all except the last 2 layers
                        device=0) # specify device=0 for gpu training