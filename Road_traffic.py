!pip install roboflow
!pip install ultralytics
from ultralytics import YOLO
from IPython import display
from IPython.display import display,Image
#display.Clear_output()
!yolo mode=Checks

from roboflow import Roboflow
rf = Roboflow(api_key="fNgnxrBW1x0fseS5yZqT")
project = rf.workspace("lr-tdx").project("road-mark")
dataset = project.version(3).download("yolov8")
!yolo task= detect mode=train model=yolov8l.pt data={dataset.location}/data.yaml epochs=20 imgsz=640
!yolo task= detect mode=val model=yolov8l.pt data={dataset.location}/data.yaml 
!yolo task= detect mode=predict model=yolov8l.pt data={dataset.location}/data.yaml 
import glob
from IPython.display import Image,display
for image_path in glob.glob (f'/content/runs/detect/predict/*.jpg'):
      display(Image(filename=image_path,height=600))
      print("\n")
  Image(filename=f'/content/runs/detect/train/confusion_matrix.png')
 Image(filename=f'/content/runs/detect/train/results.png')
