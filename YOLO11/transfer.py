from ultralytics import YOLO

""" This is an example function to use transfer learning with the ultralytics YOLO11 model. """

model = YOLO("yolo11n.yaml") # build a new model from yaml
model = YOLO("yolo11n.pt") # load pretrained model
model = YOLO("yolo11n.yaml").load("yolo11n.pt") # build from yaml and transfer weights

results = model.train(data='<datasetname>', epochs=100, imsz=640) # specify device=0 for gpu training