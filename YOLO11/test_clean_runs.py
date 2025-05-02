import shutil
import os
from YOLO11.predict import clean_runs

def test_clean_runs():
    """ test for function that clears old prediction runs"""
    testPath = "testDir"
    os.makedirs(testPath, exist_ok=True)

    clean_runs(testPath)

    if os.path.exists(testPath):
        print(f"Folder {testPath} not deleted")
    else:
      print("test_clean_runs success")