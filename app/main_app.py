import cv2
import sys
import numpy as np
import tritonclient.http as httpclient

# --- Configuration ---
VIDEO_PATH = "test_video.mp4" 
TRITON_URL = "localhost:8000"
MODEL_NAME = "yolo_model"
MODEL_INPUT_SHAPE = (640, 640) # (Height, Width)

# Post-processing thresholds
CONF_THRESHOLD = 0.5  # Confidence threshold for a detection to be considered
NMS_THRESHOLD = 0.4   # Non-Maximum Suppression threshold to remove overlapping boxes

# Define the classes your model was trained on.
# IMPORTANT: This list MUST match the order of classes from your training dataset.
# Example for a model trained on COCO's first 4 classes:
CLASSES = ["person", "bicycle", "car", "motorbike"]

def preprocess(frame, input_shape):
    """Preprocesses a frame for YOLOv8 inference."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_shape)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    return np.expand_dims(img_transposed, axis=0)

def postprocess(output_data, frame_shape, conf_threshold, nms_threshold):
    """
    Post-processes the raw output from the YOLOv8 model to get final bounding boxes.
    """
    # Unpack the frame shape
    frame_height, frame_width = frame_shape

    # The output shape is (1, 9, 8400). Transpose it to (1, 8400, 9) for easier processing.
    output_transposed = np.transpose(output_data, (0, 2, 1))[0]

    # --- 1. Filter based on confidence ---
    scores = np.max(output_transposed[:, 4:], axis=1)
    high_confidence_mask = scores > conf_threshold
    
    # Apply the mask to get only high-confidence detections
    detections = output_transposed[high_confidence_mask]
    
    if detections.shape[0] == 0:
        return [] # No detections found

    # --- 2. Prepare data for NMS ---
    # Extract boxes, scores, and class IDs
    boxes_xywh = detections[:, :4]
    class_ids = np.argmax(detections[:, 4:], axis=1)
    confidences = np.max(detections[:, 4:], axis=1)

    # Convert xywh to xyxy format required by NMS
    x_center, y_center, width, height = boxes_xywh.T
    x1 = (x_center - width / 2)
    y1 = (y_center - height / 2)
    x2 = (x_center + width / 2)
    y2 = (y_center + height / 2)
    
    # Scale coordinates from 640x640 model space to original frame space
    scale_x = frame_width / MODEL_INPUT_SHAPE[1]
    scale_y = frame_height / MODEL_INPUT_SHAPE[0]
    
    boxes_for_nms = np.array([x1 * scale_x, y1 * scale_y, width * scale_x, height * scale_y], dtype=np.float32).T

    # --- 3. Apply Non-Maximum Suppression (NMS) ---
    # This function returns the indices of the boxes to keep.
    indices = cv2.dnn.NMSBoxes(boxes_for_nms.tolist(), confidences.tolist(), conf_threshold, nms_threshold)
    
    if len(indices) == 0:
        return []

    # --- 4. Prepare final output ---
    final_detections = []
    for i in indices.flatten():
        box = boxes_for_nms[i]
        score = confidences[i]
        class_id = class_ids[i]
        final_detections.append((box.astype(int), score, class_id))
        
    return final_detections


def main():
    """Main function to capture video, run inference, and display results."""
    # --- Triton Client Setup ---
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    except Exception as e:
        print(f"Error connecting to Triton: {e}")
        sys.exit(1)

    # --- Video Capture Setup ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        sys.exit(1)

    print("Pipeline running... Press 'q' to quit.")
    
    # --- Main Inference Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame and run inference
        preprocessed_frame = preprocess(frame, MODEL_INPUT_SHAPE)
        input_tensor = httpclient.InferInput("images", preprocessed_frame.shape, "FP32")
        input_tensor.set_data_from_numpy(preprocessed_frame)

        try:
            results = triton_client.infer(model_name=MODEL_NAME, inputs=[input_tensor])
            output_data = results.as_numpy("output0")
            
            # Post-process the output to get clean detections
            detections = postprocess(output_data, frame.shape[:2], CONF_THRESHOLD, NMS_THRESHOLD)
            
            # --- Visualization ---
            for box, score, class_id in detections:
                x, y, w, h = box
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Create label text
                label = f"{CLASSES[class_id]}: {score:.2f}"
                
                # Draw background for label
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - label_height - 10), (x + label_width, y), (0, 255, 0), cv2.FILLED)
                
                # Draw label text
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        except Exception as e:
            print(f"Inference failed: {e}")
            continue

        cv2.imshow("AI Video Analytics", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    main()