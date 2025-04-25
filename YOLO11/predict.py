from ultralytics import YOLO


# Load a model
model = YOLO("weights/best.pt")  # load the best model 


# Predict with the model for video input
# results = model.predict("IMG_4400.mp4", show=True, stream=True, save=True, device=0)  # predict on an image

# predict for image
results = model.predict(source="datasets/Lightsabers.v1i.yolov11/test/images", save=True, device=0)  # predict on an image


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
# pydoc.writedoc('main')
