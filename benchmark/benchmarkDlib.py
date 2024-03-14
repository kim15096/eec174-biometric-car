import os
import argparse
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time
import dlib
import cv2

def calculate_iou(bbox1, bbox2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1 (dict): Bounding box coordinates (x1, y1, width, height).
        bbox2 (dict): Bounding box coordinates (x1, y1, width, height).

    Returns:
        float: IoU value.
    """
    x1 = max(bbox1['x1'], bbox2['x1'])
    y1 = max(bbox1['y1'], bbox2['y1'])
    x2 = min(bbox1['x1'] + bbox1['width'], bbox2['x1'] + bbox2['width'])
    y2 = min(bbox1['y1'] + bbox1['height'], bbox2['y1'] + bbox2['height'])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = bbox1['width'] * bbox1['height']
    bbox2_area = bbox2['width'] * bbox2['height']

    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    return iou

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov8n-face.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default='cpu', help='augmented inference')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save_folder', default='./WIDER_val/images/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='./WIDER_val/images/', type=str, help='dataset path')
    parser.add_argument('--gt_file', default='./benchmark/wider_face_val_bbx_gt.txt', type=str, help='ground truth file')
    opt = parser.parse_args()
    # Initialize variables for evaluation
    total_images = 0
    total_correct_detections = 0
    total_latency = 0

    annotations = []
    with open(opt.gt_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        image_info = {}
        file_name = lines[i].strip()
        # Append "WIDER_val/images" to the file name
        file_name = "./WIDER_val/images/" + file_name
        image_info['file_name'] = file_name

        i += 1  # Move to the next line containing the number of bounding boxes
        num_boxes = int(lines[i].strip())  # Parse the number of bounding boxes
        i += 1  # Move to the next line containing the first bounding box

        boxes = []
        for _ in range(num_boxes):
            box_info = lines[i].strip().split()
            x1, y1, w, h = map(int, box_info[:4])
            boxes.append({'x1': x1, 'y1': y1, 'width': w, 'height': h})
            i += 1  # Move to the next line

        image_info['bounding_boxes'] = boxes
        annotations.append(image_info)

    predictor_path = "./archive/face_detector/data/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for image_info in annotations:
        ground_truth_bboxes = image_info['bounding_boxes']
        image_path = image_info['file_name']
        img = cv2.imread(image_path)
        predicted_bboxes = []
        # Measure start time for inference
        start_time = time.time()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for i, rect in enumerate(rects):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            bbox_info = {'x1': x, 'y1': y, 'width': w, 'height': h}
            predicted_bboxes.append(bbox_info)


        # Measure end time for inference
        end_time = time.time()

        # Calculate latency for this image
        latency = end_time - start_time
        total_latency += latency

        
        for gt_bbox in ground_truth_bboxes:
            total_images += 1
            for detection in predicted_bboxes:
            # Compare detected faces with ground truth
                    iou = calculate_iou(detection, gt_bbox)
                    if iou > 0.5:  # Consider it a correct detection if IoU > 0.5
                        total_correct_detections += 1
                        break  # If a match is found, break to avoid double counting



# Calculate evaluation metrics
average_precision = total_correct_detections / total_images
average_latency = total_latency / total_images

print(f"Average Precision: {average_precision:.4f}")
print(f"Average Latency: {average_latency:.4f} seconds")
