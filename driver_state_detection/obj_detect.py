'''
Command Line Interface Help

Example: python3 yolo_img_detector.py -f imgs/beach.jpg -w yolo_files/yolov3.weights -c yolo_files/yolov3.cfg -l yolo_files/coco.names -i -ca

-f: file(s) path
--folder: folder path (can't specify both -f and --folder)
-w: weights path
-c: config path
-l: labels/names path
-i: show inference
-ca: show total classes

'''

import os
import cv2
import numpy as np

# TODO: Phase 1
import argparse
import time
import sys
import glob

probability_minimum = 0.5
threshold = 0.3

# Main function to run YOLO on images
def yolo(image, weights, config):
    # Load YOLO model and convert loaded image to blob
    network = cv2.dnn.readNetFromDarknet(config, weights)
    if network is None:
        sys.exit('Error: Invalid Weights or Config File')
    yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']
    input_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    
    # Define variables for drawing on image
    bounding_boxes = []
    confidences = []
    classes = []
    h, w = image.shape[:2]

    # Pass image through network
    network.setInput(input_blob)
    output = network.forward(yolo_layers)

    # Get bounding boxes, confidences, and classes
    for result in output:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                classes.append(class_current)
        
    return bounding_boxes, confidences, classes

# Function to draw bounding boxes and save annotated images
def save_annotated_image(image, bounding_boxes, confidences, classes, image_path, labels):
    annotated_image = image.copy()
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    
    # Get final classes and bounding boxes and confidences after NMS
    final_boxes = []
    final_confidences = []
    final_classes = []

    for i in results:
        final_boxes.append(bounding_boxes[i])
        final_confidences.append(confidences[i])
        final_classes.append(classes[i])
    
    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')
    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box = [int(j) for j in colours[classes[i]]]
            label = labels[classes[i]]
            cv2.rectangle(annotated_image, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box, 5)
            cv2.putText(annotated_image, f'{label}: {confidences[i]:.4f}', (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box, 2)
    # Save annotated image with '_out' suffix
    annotated_image_path = os.path.join('out_imgs', os.path.basename(image_path).replace('.jpg', '_out.jpg'))
    cv2.imwrite(annotated_image_path, annotated_image)
    print(f"Annotated image saved as {annotated_image_path}")

    return final_classes

def process_images(image_paths, weights, config, show_inference, show_classes_all, labels):
    # Initialize variables
    total_classes_detected = 0
    inference_times = []
    class_hash = {}
    total_counts = {}

    
    # Iterate each image path specified
    for image_path in image_paths:
        class_hash[image_path] = {}
        # Do not print if either of the inference or classes_all flag is True
        if not (show_inference or show_classes_all):
            print('-' * 50)
            print(f"Processing image: {image_path}...")
    
        image = cv2.imread(image_path)
        # Check if image_path is valid
        if image is None:
            sys.exit(f'Error: The file at {image_path} does not exist or is not a valid image.')
            
        # Start measuring time
        start_time = time.time()
        # Start the YOLO network
        bounding_boxes, confidences, classes = yolo(image, weights, config)
        # End measuring time
        end_time = time.time()
        # Get inference times
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        # Annotate image and save file
        final_classes = save_annotated_image(image, bounding_boxes, confidences, classes, image_path, labels)
        # Count for each class
        for cls in final_classes:
            class_hash[image_path][labels[cls]] = class_hash[image_path].get(labels[cls], 0) + 1
        # Get total number of classes
        total_classes_detected += len(final_classes)

    # Calculate average inference time
    average_inference_time = sum(inference_times) / len(inference_times)
    print('\n')
    print(f"{'='*22}Result{'=' * 22}")
    # If inference flag is True
    if show_inference:
        print(f"Average Inference Time: {average_inference_time:.3f} seconds")
    # If classes_all flag is True
    if show_classes_all:
        print(f"Total Number of Classes Detected: {total_classes_detected}")

    # Do not print if either of the inference or classes_all flag is True
    if not (show_inference or show_classes_all):
        print(f"Average Inference Time: {average_inference_time:.3f} seconds")
        print(f"Total Number of Classes Detected: {total_classes_detected}")

        for img, objects in class_hash.items():
            # Iterate through each object in the image
            for object_name, count in objects.items():
                # If the object is already in the total_counts dict, add the count
                if object_name in total_counts:
                    total_counts[object_name] += count
                # If not, add the object to the dict with its count
                else:
                    total_counts[object_name] = count
        
        print("\nTotal Detection Breakdown:")
        for obj_name, count in total_counts.items():
            print(f"{obj_name}: {count}")

        print("\nPer Image Breakdown:")
        for image_path in image_paths:
            print(f"{image_path} => {' | '.join([f'{cls}: {class_hash[image_path][cls]}' for cls in class_hash[image_path]])}")
    
# Run main script
if __name__ == '__main__':
    # Create command line interface for different flags
    parser = argparse.ArgumentParser(description='YOLO Image Detector')
    parser.add_argument('-f', '--files', nargs='+', help='Path to image(s)')
    parser.add_argument('--folder', help='Path to folder')
    parser.add_argument('-w', '--weights', required=True, help='Path to YOLO weights file')
    parser.add_argument('-c', '--config', required=True, help='Path to YOLO config file')
    parser.add_argument('-l', '--labels', required=True, help='Path to coco.names file')
    parser.add_argument('-i', '--inference', action='store_true', help='Show average inference time')
    parser.add_argument('-ca', '--classes_all', action='store_true', help='Show total number of classes detected')
    args = parser.parse_args()
    
    # Check if the config file exists
    if not os.path.isfile(args.config):
        print(f"Error: Configuration file {args.config} does not exist.")
        # Handle the error, for example, by exiting the program
        sys.exit(1)

    # Check if the weights file exists
    if not os.path.isfile(args.weights):
        print(f"Error: Weights file '{args.weights}' does not exist.")
        sys.exit(1)

    # Assign labels array from coco.names
    try:
        with open(args.labels, 'r') as f:
            labels = [line.strip() for line in f]
    except FileNotFoundError:
        sys.exit(f"Error: File '{args.labels}' not found")
    
    if args.folder and args.files:
        print(f"Error: Specify either file(s) with --files or folder with --folder")
        sys.exit(1)
        
    # Check if folder path exists
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"Error: Folder '{args.folder}' does not exist.")
            sys.exit(1)
        image_file_type = ('*.jpg')
        args.files = []
        args.files.extend(glob.glob(os.path.join(args.folder, image_file_type)))
        
    process_images(args.files, args.weights, args.config, args.inference, args.classes_all, labels)

    print("\nFinished!\n")



