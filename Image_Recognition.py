import numpy as np
from ultralytics import YOLO
import cv2
import math
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Load the image
img = cv2.imread("p1.jpg")

# Load the YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Class names for detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Perform object detection
results = model(img, stream=True)

# Initialize lists for detection and ground truth data
predicted_boxes = []
ground_truth_boxes = [
    [50, 50, 150, 150, "person"],
    [200, 200, 300, 300, "car"],
    [350, 350, 450, 450, "motorbike"],
    [500, 500, 600, 600, "truck"],
    [650, 650, 750, 750, "backpack"]
    # Add more ground truth boxes as required
]

# Iterate over detection results
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Bounding Box Coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])
        currentClass = classNames[cls]

        # Store predicted bounding box and class
        if conf > 0.2:  # Lowered the confidence threshold to 0.2
            predicted_boxes.append([x1, y1, x2, y2, conf, currentClass])

# Prepare data for accuracy, precision, and F1 score
y_true = []
y_pred = []

# Populate true labels from ground truth
for gt in ground_truth_boxes:
    y_true.append(gt[4])  # Ground truth class

# Populate predicted labels from detected objects
for pred in predicted_boxes:
    y_pred.append(pred[5])  # Predicted class

# Fill unmatched entries with "No Prediction" or "No Ground Truth"
max_len = max(len(y_true), len(y_pred))
y_true.extend(["No Prediction"] * (max_len - len(y_true)))
y_pred.extend(["No Ground Truth"] * (max_len - len(y_pred)))

# Ensure consistency in length
assert len(y_true) == len(y_pred), "Mismatch in lengths of y_true and y_pred"

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Visualize detected objects on the image
for detection in predicted_boxes:
    x1, y1, x2, y2, conf, currentClass = detection
    
    # Draw rectangle around detected object
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Label with class name and confidence
    label = f"{currentClass} {conf:.2f}"
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Reduce the size of the image for display
height, width = img.shape[:2]
display_width = 800  # Desired width for display (can change)
scale_factor = display_width / width
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)
img_resized = cv2.resize(img, (new_width, new_height))

# Show the result
cv2.imshow("Detected Objects", img_resized)
cv2.waitKey(0)  # Wait for the user to press a key before closing
cv2.destroyAllWindows()  # Close all OpenCV windows
