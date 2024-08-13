# Automated Box Dimension Prediction with YOLOv8 and SAM Integration

## Overview

This repository contains the code for a project focused on segmenting boxes using YOLOv8 with Segment Anything Model (SAM) integration. The goal is to precisely predict masks for the boxes and further refine the segmentation using SAM to accurately calculate the length and width of the boxes.

## Installation
1. Clone the repository:

   git clone https://github.com/a-r-p-i-t/Automated_Box_Dimension_Estimation.git

2. Install dependencies:

   pip install -r requirements.txt

## Usage
1. Train YOLOv8 segmentation model:

   ! yolo task=detect mode=train data=data.yaml model=yolov8l-seg.pt epochs=100 imgsz=640

2. Download the mobile-sam model.

3. Run main.py to generate masks over the top box top surface.
   

## Prediction Results 

![1703332856 7339106](https://github.com/a-r-p-i-t/neometry/assets/99071325/6c871434-9c84-4937-8348-b2c3a97ad718)
![1703333423 5605545](https://github.com/a-r-p-i-t/neometry/assets/99071325/298e42f8-8bc0-45cf-8509-601d80c6c962)

## Acknowledgments

1. YOLOv8: https://github.com/ultralytics/ultralytics
2. Mobile-SAM: https://github.com/ChaoningZhang/MobileSAM
   


      

