import os
import argparse
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time

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
    parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='./WIDER_val/images/', type=str, help='dataset path')
    parser.add_argument('--gt_file', default='wider_face_val_bbx_gt.txt', type=str, help='ground truth file')
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
        file_name = "WIDER_val/images/" + file_name
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

    # Initialize YOLO model
    model = YOLO(opt.weights)

    for image_info in annotations:
        ground_truth_bboxes = image_info['bounding_boxes']
        image_path = image_info['file_name']
        

        # Measure start time for inference
        start_time = time.time()

        # Run YOLO model
        results = model.predict(source=image_path, imgsz=opt.img_size, conf=opt.conf_thres, iou=opt.iou_thres, augment=opt.augment, device=opt.device)

        # Measure end time for inference
        end_time = time.time()

        # Calculate latency for this image
        latency = end_time - start_time
        total_latency += latency

        # Extract predicted bounding boxes
        result = results[0].cpu().numpy()
        predicted_bboxes = result.boxes
        predicted_bboxes = []

        for box in result.boxes:
            conf = box.conf[0]
            cls  = box.cls[0]
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = xyxy
            width = x2 - x1
            height = y2 - y1
            x = x1
            y = y1
            bbox_info = {'x1': x, 'y1': y, 'width': width, 'height': height}
            predicted_bboxes.append(bbox_info)
        
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
