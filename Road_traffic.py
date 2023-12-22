import numpy as np
import pandas as pd
from ultralytics import YOLO
model=YOLO('best(1).pt')
results = model(source=1,show=True,conf=0.3,save=True)