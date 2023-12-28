import os
import cv2
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



def infer(model,image_path):
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:-1]
    predictions = model.predict(image_path,save=False,show=False,show_labels=False,conf=0.5)
    for prediction in predictions:
        masks = prediction.masks.xy
        new_list = []
        for items in masks[0]:
            new_list.append(items[0]/img_width)
            new_list.append(items[1]/img_height)
        return new_list
    
def normalize_points(contour_points, image_width, image_height):
    # Normalize the points
    print(type(contour_points))
    normalized_points = contour_points.astype(float) / np.array([image_width, image_height])
    return normalized_points

def convert_contour_to_yolov8(contour_points, class_index):
    # Flatten the contour points to a 1D array
    flattened_points = contour_points.reshape(-1, 2)

    # Create YOLOv8 label string
    label_string = f"{class_index}"
    for point in flattened_points:
        label_string += f" {point[0]} {point[1]}"

    return label_string

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def calculate_side_centre(point1, point2):
    """
    Calculate the center point between two given points.
    """
    x = (point1[0] + point2[0]) / 2
    y = (point1[1] + point2[1]) / 2
    return (x, y)

def calculate_point_on_line(point1, point2, distance_ratio):
    """
    Calculate a point on the line between two given points based on a distance ratio.
    """
    x = point1[0] + distance_ratio * (point2[0] - point1[0])
    y = point1[1] + distance_ratio * (point2[1] - point1[1])
    return (x, y)

def find_quadrilateral_center_and_points(box_set1, distance_ratio=0.4):
    """
    Find the center point of the quadrilateral and points 20% away from it
    on the lines between side_centre1, side_centre3 and side_centre2, side_centre4.
    """
    
    p1, p2, p3, p4 = box_set1
    side_centre1 = calculate_side_centre(p1, p2)
    side_centre2 = calculate_side_centre(p2, p3)
    side_centre3 = calculate_side_centre(p3, p4)
    side_centre4 = calculate_side_centre(p4, p1)

    print("side_centre1:", side_centre1)

    quadrilateral_center = calculate_side_centre(side_centre1, side_centre3)

    

    point_on_line1 = calculate_point_on_line(side_centre1, side_centre3, distance_ratio=0.4)
    point_on_line2 = calculate_point_on_line(side_centre2, side_centre4, distance_ratio=0.4)
    point_on_line3 = calculate_point_on_line(side_centre1, side_centre3, distance_ratio=0.6)
    point_on_line4 = calculate_point_on_line(side_centre2, side_centre4, distance_ratio=0.6)
    
    return quadrilateral_center, point_on_line1, point_on_line2, point_on_line3, point_on_line4


image_dir = "/Users/uditanshusatpathy/Downloads/Neophyte/neoMetry/crop_images/1703332928.907184"
# image_files = glob.glob(image_dir +".jpg")
image_files = [image_dir +".jpg"]
save_folder= './save_txt/'
sam_checkpoint = "/Users/uditanshusatpathy/Downloads/Neophyte/neoMetry/neometry/mobile_sam.pt"
model_path = "best.pt"
model_type = "vit_t"
load_strt_time = time.time()
model = YOLO(model_path)
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(mobile_sam)
load_end_time = time.time()
print(load_end_time-load_strt_time)

device = "mps"

mobile_sam.to(device=device)

loop_strt_time = time.time()


for image_file in image_files: 
    image = cv2.imread(image_file)
    img_height, img_width = image.shape[:-1]
    img_name = os.path.basename(image_file)
    label_name = os.path.splitext(os.path.basename(image_file))[0] + ".txt"
    process_strt_time = time.time()
    seg_list = infer(model,image_file)
    # print(seg_list)

    points_set1 = np.array(seg_list).reshape(-1, 2)

    scaled_points_set1 = (points_set1 * np.array([img_width, img_height])).astype(int)

    rect_set1 = cv2.minAreaRect(scaled_points_set1)
    box_set1 = cv2.boxPoints(rect_set1).astype(int)

    for i in box_set1:
        cv2.circle(image,(i[0],i[1]), 3, (0,255,0), -1)



    center_point, point_on_line1, point_on_line2, point_on_line3, point_on_line4 = find_quadrilateral_center_and_points(box_set1)

    # print("box_set1:", box_set1)

    # print("Center point:", center_point)
    # print("Point 40% away from side_centre1 to side_centre3:", point_on_line1)
    # print("Point 40% away from side_centre2 to side_centre4:", point_on_line2)
    # print("Point 60% away from side_centre1 to side_centre3:", point_on_line3)
    # print("Point 60% away from side_centre2 to side_centre4:", point_on_line4)
    
    
    predictor.set_image(image)
    points_list=[center_point, point_on_line1, point_on_line2, point_on_line3, point_on_line4]
    label_list=[1,1,1,1,1]
    label_string_list= []
    input_point = np.array(points_list)
    input_label = np.array(label_list)
    masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
                )
    # print(masks)
    contour, _ = cv2.findContours(masks[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_list = []
    count_countour = 0
    while count_countour < len(contour):
        area = cv2.contourArea(contour[count_countour])
        area_list.append(area)
        count_countour += 1
    
    area_index = area_list.index(max(area_list))
    normalized_points = normalize_points(contour[area_index], img_width, img_height)
    label_string = convert_contour_to_yolov8(normalized_points, class_index=0)
    label_string_list.append(label_string)
    process_end_time = time.time()
    process_fps = 1/(process_end_time-process_strt_time)
    print(process_fps)
    with open(os.path.join(save_folder, label_name), 'w') as file:
                # Write each element of the list on a new line
        for item in label_string_list:
            file.write(str(item) + '\n')




    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig(f"saved_masks/{img_name}")
    plt.close()


    
loop_end_time = time.time()
total_time = loop_end_time-loop_strt_time
fps = len(image_files)/total_time
print(fps)

