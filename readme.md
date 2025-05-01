install python virtual environment
use pip install ultralytics
add a folder called "weights" to the YOLO11 folder
download best.7z model into weights directory and unpack
to use prediction, run python predict.py in the YOLO11 directory
to train your own model, run python train.py after downloading the dataset from roboflow universe.

If you have issues with ultralytics not finding the folders, ensure that your paths in ultralytics settings.json file are set correctly. 


using pydoc:
pydoc is already included in standard libraries. Make sure to comment code using """ documentation here """.

to use in a module 
import pydoc

pydoc.writedoc('name_of_file')

to use in CLI
python -m pydoc -w <name_of_file>

