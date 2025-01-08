import os
import cv2
import torch
import numpy as np
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.ops import nms
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

# Class label mapping
class_mapping = {1: "with_mask", 2: "without_mask", 3: "mask_weared_incorrect"}

def preprocess_frame(frame, target_size=(320, 320)):
    """
    Preprocesses a single frame for object detection.
    """
    original_height, original_width = frame.shape[:2]

    # Resize the frame
    resized_frame = cv2.resize(frame, target_size)
    resize_ratio_x = target_size[0] / original_width
    resize_ratio_y = target_size[1] / original_height

    # Normalize the frame
    frame_tensor = torch.tensor(resized_frame / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    return frame_tensor, resize_ratio_x, resize_ratio_y

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping boxes.
    """
    indices = nms(boxes, scores, iou_threshold)
    return indices

def draw_predictions(frame, outputs, resize_ratio_x, resize_ratio_y, confidence_threshold=0.5, iou_threshold=0.5):
    """
    Draws predictions on the frame, including NMS filtering.
    """
    boxes = outputs["boxes"]
    scores = outputs["scores"]
    labels = outputs["labels"]

    # Filter predictions by confidence threshold
    keep = scores > confidence_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Apply NMS
    keep_indices = apply_nms(boxes, scores, iou_threshold)
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    # Draw filtered predictions
    for box, label in zip(boxes, labels):
        # Scale the bounding box back to the original frame size
        xmin, ymin, xmax, ymax = map(int, box)
        xmin = int(xmin / resize_ratio_x)
        ymin = int(ymin / resize_ratio_y)
        xmax = int(xmax / resize_ratio_x)
        ymax = int(ymax / resize_ratio_y)

        label_text = class_mapping.get(label.item(), "unknown")
        # Draw bounding box and label
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)  # Red bounding box
        cv2.putText(
            frame,
            label_text,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,  # Larger font size
            (0, 0, 255),  # Red color for text
            3  # Text thickness
        )
    return frame

def real_time_object_detection(model, confidence_threshold=0.5, iou_threshold=0.5, camera_index=0):
    """
    Launches a real-time object detection application using the webcam.
    """
    # Initialize the webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess the frame
        frame_tensor, resize_ratio_x, resize_ratio_y = preprocess_frame(frame)

        # Perform inference
        with torch.no_grad():
            outputs = model(frame_tensor)[0]

        # Draw predictions on the frame
        frame_with_predictions = draw_predictions(frame, outputs, resize_ratio_x, resize_ratio_y, confidence_threshold, iou_threshold)

        # Display the frame
        cv2.imshow("Real-Time Object Detection", frame_with_predictions)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Define the model
model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)  
num_classes = 4
model.head.classification_head.num_classes = num_classes

# Load the trained weights
model.load_state_dict(torch.load("trained_ssd_model/ssd_mobilenet_trained.pth", map_location=torch.device("cpu"), weights_only=True))
model.to("cpu")  # Ensure the model is on CPU
model.eval()

# Start real-time object detection
real_time_object_detection(model, confidence_threshold=0.5, iou_threshold=0.5, camera_index=1)
