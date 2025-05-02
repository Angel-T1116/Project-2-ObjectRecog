# Object Recognition System - ECEN 4273/5080 Project 2

## Authors
Tyler Curtis <br>  
Cosette Byte <br>  
Angel Trujillo <br>


## Overview
This project implements an object recognition system using deep learning to detect and annotate specific objects within video frames. The project uses Ultralytics YOLOv11 to train and predict. The system can recognize the following objects with at least 75% accuracy in most cases:
- Dalek
- Lightsaber
- Cat
- Dog
- Person

The input is a video file (.mp4) or an image (jpeg, jpg, png), and the output is a new video file (.mp4) or image with bounding boxes and labels around detected objects. <br> 
Included in the repository are the image scrapers used to collect the dataset

---

## Features
- Deep learning-based object detection
- Supports inference on video files
- Outputs videos with bounding boxes and labels
- Achieves greater than 75% accuracy on test datasets
- Capable of detecting multiple objects per frame

---

## Getting Started
It is recommended to use this project in a linux environment. For Windows users we recommend setting up wsl. <br>  

### 1. Dependencies
In order to use the project you must have the following installed: <br>  
Python 3.8+ <br>  



### 2. Installation
Clone the repository:
gh repo clone Angel-T1116/Project-2-ObjectRecog <br>  
navigate to the repository on your local machine. <br>  


### 3. Create a Python Virtual Environment and activate it
- python -m venv venv
- venv\Scripts\activate

### 4. Install Required Packages
pip install -r requirements.txt

### 5. Usage
- add a folder called "weights" to the YOLO11 folder
- download best.7z model into weights directory and unpack to use prediction
- run python predict.py in the YOLO11 directory to train your own model
- run python train.py after downloading the dataset from roboflow universe to train your own model.

---

## Documentation
Documentation for this project is automatically generated using pydoc and hosted on GitHub Pages: To generate documentation locally use
- python -m pydoc -w YOLO11.predict
- python -m pydoc -w YOLO11.test_clean_runs
- python -m pydoc -w YOLO11.test_video_converter
- python -m pydoc -w YOLO11.train

---

## CI/CD
This project uses GitHub Actions for Continuous Integration (CI). On each push to main, the workflow:
- Installs dependencies
- Lints code with flake8
- Runs all tests with pytest
- Generates pydoc documentation
- Pushes updates to /documentation folder
Workflow file: .github/workflows/python-app.yml

### Common Issues
If you have issues with ultralytics not finding the folders, ensure that your paths in ultralytics settings.json file are set correctly. <br>  

When training with a GPU, you may encounter issues with the version of CUDA on your local machine and pytorch. In this case, remove pytorch from your virtual environment and visit https://pytorch.org/get-started/locally/ to get the correct version of pytorch for your CUDA install. <br>  
