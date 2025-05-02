# Object Recognition System - ECEN 4273/5080 Project 2

## Authors
Angel Trujillo
Tyler Curtis
Cosette Byte

## Overview
This project implements an object recognition system using deep learning to detect and annotate specific objects within video frames. The system can recognize the following objects with at least 75% accuracy:
- Dalek
- Lightsaber
- Cat
- Dog
- Person

The input is a video file (.mp4) or an image (jpeg, jpg, png), and the output is a new video file (.mp4) or image with bounding boxes and labels around detected objects.

---

## Features
- Deep learning-based object detection
- Supports inference on video files
- Outputs videos with bounding boxes and labels
- Achieves greater than 75% accuracy on test datasets
- Capable of detecting multiple objects per frame

---

## Getting Started

### 1. Dependencies
Install the required libraries:
pip install:
Python 3.8+
PyTorch / TensorFlow
OpenCV
Ultralytics
Pydoc

### 2. Installation
Clone the repository:
git clone https://github.com/yourusername/object-recognition-project.git
cd object-recognition-project

### 3. Create a Python Virtual Environment and activate it
- python -m venv venv
- venv\Scripts\activate

### 4. Install Required Packages
pip install -r requirements.txt

### 5. Usage
add a folder called "weights" to the YOLO11 folder download best.7z model into weights directory and unpack to use prediction, run python predict.py in the YOLO11 directory to train your own model, run python train.py after downloading the dataset from roboflow universe.

To annotate a video:
python annotate_video.py --input path/to/input_video.mp4 --output path/to/output_video.mp4

### 6. Running the Prediction
python predict.py

---

## Documentation
Documentation for this project is automatically generated using pydoc and hosted on GitHub Pages: To generate documentation locally use
python -m pydoc -w predict

Using pydoc: pydoc is already included in standard libraries. Make sure to comment code using the syntax """ documentation here """

to use in a module import pydoc:
pydoc.writedoc('name_of_file')
to use in CLI python -m pydoc -w <name_of_file>

---

## CI/CD
This project uses GitHub Actions for Continuous Integration (CI). On each push to main, the workflow:
- Installs dependencies
- Lints code with flake8
- Runs all tests with pytest
- Generates pydoc documentation
- Pushes updates to /docs folder (auto-deployed by GitHub Pages)
Workflow file: .github/workflows/python-app.yml

### Common Issues
If you have issues with ultralytics not finding the folders, ensure that your paths in ultralytics settings.json file are set correctly.
