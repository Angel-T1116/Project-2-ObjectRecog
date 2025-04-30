import shutil
import os
from predict import clean_runs

def test_clean_runs():
    testPath = "testDir"
    os.mkdir(testPath, exist_ok=True)

    clean_runs(testPath)

    if os.path.exists(testPath):
        print(f"Folder {testPath} not deleted")
    else:
      print("test_clean_runs success")