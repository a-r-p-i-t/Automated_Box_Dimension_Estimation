# Automated Box Dimension Prediction with YOLOv8 and SAM Integration

## Overview

This project focuses on accurately predicting the dimensions (length, breadth, and height) of boxes using advanced segmentation techniques using YOLO and SAM and depth information from a real sense camera. The project integrates machine learning models to segment the top surface of the box and converts the segmented pixels into real-world measurements.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [WorkFlow](#workflow)
- [Key Features](#key-features)
- [Results](#results)
- [Sample Segmentation Results through SAM Promts](#sample-segmentation-results-through-sam-promts)
- [Acknowledgments](#acknowledgments)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/a-r-p-i-t/Automated_Box_Dimension_Estimation.git

3. Install dependencies:

   ```bash
   pip install -r requirements.txt

### Usage

1. Train YOLOv8 segmentation model:

   ```bash
   ! yolo task=detect mode=train data=data.yaml model=yolov8l-seg.pt epochs=100 imgsz=640

2. Download the mobile-sam model and clone the Mobile SAM Repository:

   ```bash
   pip install git+https://github.com/ChaoningZhang/MobileSAM.git

4. Run **[main.py](./main.py)** to generate masks over the top box top surface via mobile SAM Model.

5. Run **[pixels_to_mm.py](./pixels_to_mm.py)** to convert pixel coordinates from the segmentation mask into real-world measurements.

6. Review the generated Excel sheet to compare predicted dimensions against ground truth measurements.

### Workflow

## 1. **Data Collection**
   - Images of boxes were captured alongside depth data using a real sense camera.
   - Ground truth measurements of the box dimensions were recorded to serve as a benchmark.

## 2. **Model Training and Inference**

   - **YOLO Model:** A custom YOLO model was trained for the initial segmentation of the box. Despite its effectiveness in general object detection, the YOLO model did not consistently segment the entire top surface of the box accurately.

   - **Segment Anything Model (SAM):** To address the shortcomings of YOLO, the Segment Anything Model (SAM) was employed. Key points were extracted from the YOLO segmentation mask to guide SAM in generating more accurate segmentations.

## 3. **Point Extraction and Prompt Generation**

   - **Point Calculation:** Four corner points of the box were extracted from the YOLO segmentation mask. These corner points were then used to calculate the side center points of the box.

   - **Prompt Generation for SAM:**
     - From the corner and side center points, 19 prompts were generated: 17 positive prompts (indicating areas to include in the segmentation) and 2 negative prompts (indicating areas to exclude).
     - The positive prompts focused on the top surface of the box, while the negative prompts helped SAM differentiate the box surface from the surrounding background.
     - SAM utilized these prompts to accurately segment the top surface of the box, leading to precise length and width measurements.

## 4. **Dimension Calculation**

   - **Segmentation Output Processing:** Once SAM completed the segmentation, the pixel coordinates of the segmented top surface were extracted.

   - **Conversion to Real-World Coordinates:**
     - The pixel coordinates were then converted into real-world coordinates using camera calibration parameters, specifically the focal length and optical center of the camera. The code provided ensures accurate transformation by adjusting the pixel positions based on the depth information from the real sense camera.
       
     - **Camera Calibration Parameters:**
       - Focal lengths (`fx`, `fy`) and optical centers (`cx`, `cy`) were used to convert the 2D pixel coordinates into 3D real-world coordinates.
       - The depth of the box, obtained from the real sense camera, played a crucial role in this conversion by determining the distance of each point from the camera, allowing for accurate scaling of the 2D coordinates to real-world dimensions.
       - Each point on the segmented top surface is initially in 2D pixel coordinates. To convert these into real-world 3D coordinates (X, Y, Z), the depth information is used in conjunction with the camera's intrinsic parameters (focal lengths fx, fy and optical centers cx, cy).
       - The conversion formula is as follows:
            -X = (x−cx)×depth/fx
            -Y = (y−cy)×depth/fy
      - Here, x and y are the pixel coordinates of a point on the image, depth is the distance from the camera to that point, and X, Y are the real-world coordinates in millimeters.
    
      - Once the 3D coordinates of all key points on the top surface are obtained using the depth data, the real-world distances (in millimeters) between these points are calculated.
      - For example, to find the length and width of the box, the Euclidean distance formula is used.
      - These distances represent the actual length and width of the box in the real world.
​

         

   - **Real-World Dimension Calculation:** Using these real-world coordinates, the algorithm calculated the dimensions (length and width) by determining the distances between key points on the segmented surface.

   - **Excel Sheet Generation:** The calculated dimensions, along with the corresponding ground truth measurements, were saved in an Excel sheet. This sheet provides a clear comparison between the predicted and actual dimensions, allowing for easy benchmarking and validation.

## 5. **Benchmarking and Validation**

   - The predicted dimensions were benchmarked against the actual, measured dimensions of the boxes.
   - The results showed that the combined YOLO and SAM approach provided high accuracy, with the predictions closely matching the ground truth data.
   - The Excel sheet generated during the process contains both the real dimensions and the predicted dimensions, facilitating a detailed comparison.

### Key Features

- **Custom YOLO Model:** Used for initial box segmentation.
- **Segment Anything Model (SAM):** Employed for refined and accurate segmentation of the box's top surface.
- **Real-World Conversion:** Pixel coordinates from the segmentation mask were accurately converted into real-world measurements.
- **Benchmarking:** Validation of the model's predictions against ground truth measurements confirmed the approach's accuracy.
- **Excel Sheet Generation:** An Excel sheet is generated containing the predicted and actual dimensions, aiding in the evaluation and presentation of results.
   

### Results 

The project successfully demonstrated that integrating a custom YOLO model with the Segment Anything Model (SAM) significantly improved the accuracy of box dimension prediction. The results were closely aligned with the ground truth measurements, showcasing the effectiveness of the developed approach.
Thus we make Box Dimension estimation Automated.

### Sample Segmentation Results through SAM Promts

![1703332856 7339106](https://github.com/a-r-p-i-t/neometry/assets/99071325/6c871434-9c84-4937-8348-b2c3a97ad718)

### Acknowledgments

1. YOLOv8: https://github.com/ultralytics/ultralytics
2. Mobile-SAM: https://github.com/ChaoningZhang/MobileSAM
   


      

