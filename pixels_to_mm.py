import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2

file_path = "D14_2024-01-08_18-42-57_24points\\results.csv"

df = pd.read_csv(file_path)



# camera focus parameters 
cx,cy = 645.256, 372.084
fx,fy = 888.660, 888.659   

def get_xy_cordinate(d_center, cordinate):
    """Thanking Ritesh Kumaar
    This function with return real world coordinate... 
    
    Args:
            d_center (_type_): _description_
            cordinate (_type_): .tolist()_description_
    
    Returns:
            _type_: _description_
    """
    x, y = cordinate[0], cordinate[1]
    
    # camera focus parameters 
    cx,cy = 645.256, 372.084
    fx,fy = 888.660, 888.659   
    
    x = x-cx
    y = y-cy
    X, Y = (d_center*x/fx, d_center*y/fy)
    return X, Y

img_width = 600
img_height = 325
count=0
for index, row in df.iterrows():
    count+=1
    print(count)
    print(len(row["predicted_masks"].split(' ')[1:]))

    # predicted_mask = [float(value) for value in row["predicted_masks"].split(" ")[1:]]
    predicted_mask_unformatted = [value for value in row["predicted_masks"].split(' ')[1:]]
    predicted_mask_formatted = []
    for i in predicted_mask_unformatted:
        if "\n" in i:
            predicted_mask_formatted.append(i.split("\n")[0])
        else:
            predicted_mask_formatted.append(i)

    print(predicted_mask_formatted)

    predicted_mask = [float(value) for value in predicted_mask_formatted]
    # ground_truth_mask = [float(value) for value in row["ground_truth_masks"].split(' ')[1:]]
    depth_mm = row["cam_box_dist"]
    box_height = row["width_mm"]
    

    
    # Create a blank image
    # image = np.zeros((img_height, img_width))

    # Reshape points into pairs of (x, y)
    points_set1 = np.array(predicted_mask).reshape(-1, 2)
    # points_set2 = np.array(ground_truth_mask).reshape(-1, 2)
    
    # Scale points to image size
    scaled_points_set1 = (points_set1 * np.array([img_width, img_height])).astype(int)
    # scaled_points_set2 = (points_set2 * np.array([img_width, img_height])).astype(int)
    
    # Create a polygon mask for Set 1 with 50% transparency
    polygon_set1 = Polygon(scaled_points_set1, edgecolor='none', facecolor=(1, 0, 0, 0.5))
    plt.gca().add_patch(polygon_set1)

    # Calculate the minimum area rectangle for Set 1
    rect_set1 = cv2.minAreaRect(scaled_points_set1)
    box_set1 = cv2.boxPoints(rect_set1).astype(int)

    x = []
    y = []
    for xy in box_set1:
        X, Y = get_xy_cordinate(depth_mm, xy)
        x.append(X)
        y.append(Y)

    d = [np.sqrt((x[i + 1] - x[i])**2 + (y[i + 1] - y[i])**2)
         for i in range(len(x) - 1)]
    d.append(np.sqrt((x[0] - x[3])**2 + (y[0] - y[3])**2))
    # calculate the pixel length and width from detected bbox
    # length_mm = (d[0] + d[2]) / 2
    # width_mm = (d[1] + d[3]) / 2
    length_mm = min(d[0] , d[2]) 
    width_mm = min(d[1] , d[3]) 
    final_length_mm = max(length_mm,width_mm)
    final_width_mm = min(length_mm,width_mm)

    df.at[index, "Predicted length (mm)"] = final_length_mm
    df.at[index, "Predicted width (mm)"] = final_width_mm
    


    # Draw the rectangle for Set 1 on the image
    # cv2.drawContours(image, [box_set1], 0, (255, 255, 255), 2)
    # cv2.imshow("image.jpg",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Display the information on the image
    # info_set1 = f"Predicted_mask :Rectangle - Length: {length_mm:.2f}, Width: {width_mm:.2f}, Height: {box_height:.2f} mm"
    # plt.text(10, 40, info_set1, color='white', fontsize=8, backgroundcolor='black')
        
    # # Create a polygon mask for Set 2 with 50% transparency
    # polygon_set2 = Polygon(scaled_points_set2, edgecolor='none', facecolor=(0, 1, 0, 0.5))
    # plt.gca().add_patch(polygon_set2)

    
    # Calculate the minimum area rectangle for Set 1
    # rect_set1 = cv2.minAreaRect(scaled_points_set2)
    # box_set1 = cv2.boxPoints(rect_set1).astype(int)

    # x = []
    # y = []
    # for xy in box_set1:
    #     X, Y = get_xy_cordinate(depth_mm, xy)
    #     x.append(X)
    #     y.append(Y)

    # d = [np.sqrt((x[i + 1] - x[i])**2 + (y[i + 1] - y[i])**2)
    #      for i in range(len(x) - 1)]
    # d.append(np.sqrt((x[0] - x[3])**2 + (y[0] - y[3])**2))
    # # calculate the pixel length and width from detected bbox
    # # length_mm = (d[0] + d[2]) / 2
    # # width_mm = (d[1] + d[3]) / 2
    # length_mm = min(d[0] , d[2]) 
    # width_mm = min(d[1] , d[3]) 

    # df.at[index, "Ground_truth length (mm)"] = length_mm
    # df.at[index, "Ground_truth width (mm)"] = width_mm
    
    # Draw the rectangle for Set 1 on the image
    # cv2.drawContours(image, [box_set1], 0, (255, 255, 255), 2)
    # cv2.imshow("image2.jpg",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # # Display the information on the image
    # info_set1 = f"ground_truth_mask :Rectangle - Length: {length_mm:.2f}, Width: {width_mm:.2f}, Height: {box_height:.2f} mm"
    # plt.text(10, 100, info_set1, color='white', fontsize=8, backgroundcolor='black')
        
    # if index < 20:
    #     # Display the image with the masks
    #     plt.imshow(image)
    #     plt.axis('off')
    #     plt.show()
    # else:
    #     break



print(df)

file_path = "pixel_mm_5_modified.xlsx"
df.to_excel(file_path, index=False)