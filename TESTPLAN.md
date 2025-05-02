# Test Plan – ECEN 4273/5080 Project 2
### Object Detection using YOLOv11

---

## 1. Purpose
This document outlines the unit and integration testing strategies used to verify the correctness and reliability of the object detection system developed for Project 2. It includes manual and automated tests applied to individual modules as well as end-to-end evaluations with sample videos and dataset inputs.

---

## 2. Scope
The tests cover:
- Unit tests for individual components
- Integration tests combining prediction and output pipelines
- Functional validation using labeled datasets and videos
- CI-based checks using GitHub Actions

---

## 3. Unit Testing

| Module                  | Test Cases                                                                   | Status |
|-------------------------|------------------------------------------------------------------------------|--------|
| `CustomVideoConverter()`| - Converts `.avi` to `.mp4` correctly<br>- Handles invalid input paths       | Passed |
| `clean_runs()`          | - Deletes old output folders/files<br>- Does not crash if folders missing    | Passed |

Tests are implemented in `test_video_converter.py` and `test_clean_runs.py`. Executed using `pytest` locally and in CI.

---

## 4. Integration Testing

| Component                | Test Description                                                                 |
|--------------------------|----------------------------------------------------------------------------------|
| YOLO Prediction Pipeline | - Runs `predict.py` on test videos<br>- Detects and labels objects<br>- Saves results |
| Combined Pipeline        | - Tests full flow: load → predict → convert video → output                      |

Manual validation was performed using multiple `.mp4` inputs. Output verified via bounding boxes and conversion.

---

## 5. Functional/Model Testing

The trained YOLOv11 model was tested on:
- Dataset samples for: Daleks, lightsabers, cats, dogs, people
- At least one test video per class
- Confirmed >75% accuracy on all categories

---

## 6. CI/CD Testing

GitHub Actions workflow performs the following:
- Dependency installation
- Code linting (`flake8`)
- Unit test execution (`pytest`)
- PyDoc HTML generation (`pydoc`)
- Auto-push documentation to GitHub Pages

---

## 7. Future Testing Ideas

- Compare frame-by-frame prediction against labeled data
- Measure inference time for video streams
- Validate real-time webcam input (if expanded)