from ultralytics import YOLO
import cv2
import os

# Define paths
MODEL_PATH = "yolo11m.pt"  # Ensure this model file is available
IMAGE_DIR = "tiles"
OUTPUT_DIR = "detections"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the YOLOv11 model
model = YOLO(MODEL_PATH)

# Process each image in the directory
for image_file in os.listdir(IMAGE_DIR):
    if image_file.endswith(".png"):
        img_path = os.path.join(IMAGE_DIR, image_file)
        image = cv2.imread(img_path)

        # Perform object detection
        results = model.predict(source=image, save=False)

        # Draw bounding boxes on the image
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the annotated image
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_file)[0]}_detected.png")
        cv2.imwrite(output_path, image)
