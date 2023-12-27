# -- coding: utf-8 --
'''
Created on 26-12-2023 19:00
Project : neometry
@author : Uditanshu Satpathy
@emails : uditanshusatpathy23@gmail.com
'''
import numpy as np

class SAM_FILTER:
    def __init__(self):
        pass

    def calculate_iou(self, contour1, contour2):
        def polygon_area(polygon):
            n = len(polygon)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += polygon[i][0] * polygon[j][1]
                area -= polygon[j][0] * polygon[i][1]
            area = abs(area) / 2.0
            return area

        def intersection_area(polygon1, polygon2):
            def line_intersection(line1, line2):
                xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
                ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

                def det(a, b):
                    return a[0] * b[1] - a[1] * b[0]

                div = det(xdiff, ydiff)
                if div == 0:
                    return None  # Lines don't intersect

                d = (det(*line1), det(*line2))
                x = det(d, xdiff) / div
                y = det(d, ydiff) / div
                return x, y

            intersection_points = []
            for i in range(len(polygon1)):
                for j in range(len(polygon2)):
                    line1 = [polygon1[i], polygon1[(i + 1) % len(polygon1)]]
                    line2 = [polygon2[j], polygon2[(j + 1) % len(polygon2)]]
                    intersection = line_intersection(line1, line2)
                    if intersection:
                        intersection_points.append(intersection)

            return polygon_area(intersection_points)
        area_contour1 = polygon_area(contour1)
        area_contour2 = polygon_area(contour2)
        intersection = intersection_area(contour1, contour2)
        iou = ((area_contour1 + area_contour2) - intersection) / intersection
        return abs(iou)
    
    def convert_normalized_to_image_coordinates(normalized_points, image):
        height, width, _ = image.shape
        return np.squeeze((normalized_points * [width, height])).astype(int)