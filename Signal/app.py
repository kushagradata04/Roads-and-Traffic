from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os

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
    vehicle_counts = []
    
    for i in range(1, 5):
        file_key = f'file{i}'
        file = request.files.get(file_key)

        if not file or file.filename == '':
            return redirect(request.url)
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Process the uploaded image and get the vehicle count
        output_image_path, vehicle_count = process_image(file_path)

        # Store the output path and count for each image
        output_images.append(output_image_path)
        vehicle_counts.append(vehicle_count)

    # Pass the vehicle counts and output images to the result template
    return render_template('result.html', output_image1=output_images[0], vehicle_count1=vehicle_counts[0],
                           output_image2=output_images[1], vehicle_count2=vehicle_counts[1],
                           output_image3=output_images[2], vehicle_count3=vehicle_counts[2],
                           output_image4=output_images[3], vehicle_count4=vehicle_counts[3])

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

    vehicle_count = 0
    boxes = []  # List to hold all bounding boxes and their confidences

    # Iterate through each detection
    for detection in detections:
        for obj in detection:
            # Extract class ID and confidence from detection
            scores = obj[5:]  # Assuming obj contains 4 coordinates + objectness score + class scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Check confidence threshold
                label = classes[class_id]
                if label in ['car', 'bus', 'truck']:
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
        
        boxes = [box for box in boxes if calculate_iou(current_box[:4], box[:4]) <= 0.8]

    # Draw remaining bounding boxes on the image
    for (x, y, w, h, confidence, label) in selected_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # Save the output image with detected bounding boxes
    output_image_path = os.path.join(OUTPUT_FOLDER, 'output_' + os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)

    return output_image_path, len(selected_boxes)

# Serve output images
@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
