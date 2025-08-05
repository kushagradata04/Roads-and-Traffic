from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import math

app = Flask(__name__)

# Load YOLO pre-trained model for vehicle detection
net = cv2.dnn.readNet("yolov7.weights", "yolov7.cfg")

# Load the COCO names file (contains class names)
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Paths
UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'output_images/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    output_images = []
    vehicle_counts_list = []
    
    for i in range(1, 5):
        file_key = f'file{i}'
        file = request.files.get(file_key)

        if not file or file.filename == '':
            return redirect(request.url)
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Process the uploaded image and get the vehicle count
        output_image_path, vehicle_counts = process_image(file_path)

        # Store the output path and count for each image
        output_images.append(output_image_path)
        vehicle_counts_list.append(vehicle_counts)

    green_light_durations = calculate_green_light_duration(vehicle_counts_list)

    # Pass the vehicle counts and output images to the result template
    return render_template('result.html', output_image1=output_images[0], vehicle_counts1=vehicle_counts_list[0],
                           output_image2=output_images[1], vehicle_counts2=vehicle_counts_list[1],
                           output_image3=output_images[2], vehicle_counts3=vehicle_counts_list[2],
                           output_image4=output_images[3], vehicle_counts4=vehicle_counts_list[3],
                           green_light_durations=green_light_durations)

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Boxes are defined by (x, y, width, height)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # Calculate the area of intersection rectangle
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Calculate the area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Calculate the intersection over union (IoU)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    
    return iou

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names and perform the forward pass
    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)

    # Initialize counters for each vehicle type
    vehicle_counts = {
        'car': 0,
        'bus': 0,
        'truck': 0,
        'motorbike': 0
    }

    boxes = []  # List to hold all bounding boxes and their confidences

    # Iterate through each detection
    for detection in detections:
        for obj in detection:
            # Extract class ID and confidence from detection
            scores = obj[5:]  # Assuming obj contains 4 coordinates + objectness score + class scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Increase the confidence threshold to reduce false positives
            if confidence > 0.4:  
                label = classes[class_id]
                if label in vehicle_counts:  # Check if the label is a vehicle type we're interested in
                    # Get bounding box coordinates relative to the image size
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append((x, y, w, h, confidence, label))  # Append bounding box with confidence and label

    # Sort boxes by confidence score in descending order
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)

    # Perform Non-Maximum Suppression (NMS) to remove overlapping boxes
    selected_boxes = []
    while boxes:
        # Select the box with the highest confidence and remove it from the list
        current_box = boxes.pop(0)
        selected_boxes.append(current_box)
        
        # Increase the IoU threshold to 0.9 for better filtering
        boxes = [box for box in boxes if calculate_iou(current_box[:4], box[:4]) <= 0.8]

    # Update the counts based on NMS results
    for (x, y, w, h, confidence, label) in selected_boxes:
        vehicle_counts[label] += 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # Save the output image with detected bounding boxes
    output_image_path = os.path.join(OUTPUT_FOLDER, 'output_' + os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)

    return output_image_path, vehicle_counts

def calculate_green_light_duration(vehicle_counts_list):
    base_time_per_ecu = 5  # seconds per ECU
    max_cycle_time = 150   # maximum total cycle time in seconds
    min_green_time = 10    # minimum green light time in seconds

    ecus_per_side = []
    ecu_weights = {'car': 1, 'motorbike': 0.5, 'bus': 2.5, 'truck': 3}

    for vehicle_counts in vehicle_counts_list:
        total_ecu = sum(vehicle_counts[vehicle] * ecu_weights[vehicle] for vehicle in vehicle_counts)
        ecus_per_side.append(total_ecu)

    # Calculate initial green times
    initial_green_times = [ecu * base_time_per_ecu for ecu in ecus_per_side]
    total_initial_time = sum(initial_green_times)

    # Exponential scaling factor
    def apply_exponential_scaling(time, index, total_count):
        # Apply exponential scaling
        scaling = 1 - (index / total_count)
        scaled_time = time * (0.5 ** scaling)
        return max(min_green_time, scaled_time)

    if total_initial_time > max_cycle_time:
        scaling_factor = max_cycle_time / total_initial_time
        green_light_durations = [apply_exponential_scaling(time, i, len(initial_green_times)) * scaling_factor for i, time in enumerate(initial_green_times)]
    else:
        green_light_durations = [apply_exponential_scaling(time, i, len(initial_green_times)) for i, time in enumerate(initial_green_times)]

    return green_light_durations
    

# Serve output images
@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
