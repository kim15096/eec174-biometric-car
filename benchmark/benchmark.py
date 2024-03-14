import mediapipe as mp
import time

      
def read_wider_face_annotations(annotations_file):
    """
    Reads the ground truth annotations from the WIDER FACE dataset.

    Args:
        annotations_file (str): Path to the ground truth text file.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary represents an image's annotations.
    """
    annotations = []
    with open(annotations_file, 'r') as f:
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

    return annotations


def calculate_iou(bbox1, bbox2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Returns:
        float: IoU value.
    """
    x1 = max(bbox1.origin_x, bbox2['x1'])
    y1 = max(bbox1.origin_y, bbox2['y1'])
    x2 = min(bbox1.origin_x + bbox1.width, bbox2['x1'] + bbox2['width'])
    y2 = min(bbox1.origin_y + bbox1.height, bbox2['y1'] + bbox2['height'])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = bbox1.width * bbox1.height
    bbox2_area = bbox2['width'] * bbox2['height']

    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

def evaluate_face_detection(detector, annotations):
    total_images = len(annotations)
    total_correct_detections = 0
    total_latency = 0  # Initialize total latency counter

    for image_info in annotations:
        ground_truth_bboxes = image_info['bounding_boxes']
        image_path = image_info['file_name']
        mp_image = mp.Image.create_from_file(image_path)

        # Measure start time for face detection
        start_time = time.time()

        # Run face detection
        face_detector_result = detector.detect(mp_image)
        
        # Measure end time for face detection
        end_time = time.time()

        # Calculate latency for this image
        latency = end_time - start_time
        total_latency += latency

        for detection in face_detector_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            # Compare detected faces with ground truth
            for gt_bbox in ground_truth_bboxes:
                iou = calculate_iou(bbox, gt_bbox)
                if iou > 0.5:  # Consider it a correct detection if IoU > 0.5
                    total_correct_detections += 1
                    break  # If a match is found, break to avoid double counting

    average_precision = total_correct_detections / total_images
    average_latency = total_latency / total_images  # Calculate average latency
    return average_precision, average_latency

if __name__ == "__main__":
    annotations_file_path = 'wider_face_val_bbx_gt.txt'
    wider_face_annotations = read_wider_face_annotations(annotations_file_path)
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)
    # Evaluate performance
    with FaceDetector.create_from_options(options) as detector:
        ap, latency = evaluate_face_detection(detector, wider_face_annotations)
        print(f"Average Precision: {ap:.4f}")
        print(f"Average Latency: {latency:.4f} seconds")