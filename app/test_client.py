import numpy as np
import tritonclient.http as httpclient
from PIL import Image

# --- Configuration ---
MODEL_NAME = "yolo_model"
URL = "localhost:8000"
IMAGE_INPUT_SHAPE = (1, 3, 640, 640) # Batch size 1, 3 channels, 640x640

print(f"Connecting to Triton server at {URL}...")

try:
    # Create a client object to connect to Triton
    client = httpclient.InferenceServerClient(url=URL)
except Exception as e:
    print(f"Client creation failed: {e}")
    exit(1)


# Create dummy input data. A real application would load and preprocess an image.
dummy_image = np.random.rand(*IMAGE_INPUT_SHAPE).astype(np.float32)

# Set up the input tensor
input_tensor = httpclient.InferInput("images", dummy_image.shape, "FP32")
input_tensor.set_data_from_numpy(dummy_image)

print(f"Sending request for model '{MODEL_NAME}'...")

# Send the request for inference
try:
    results = client.infer(
        model_name=MODEL_NAME,
        inputs=[input_tensor]
    )

    # Get the output tensor
    output_data = results.as_numpy("output0")

    print("\nSUCCESS! Successfully received response from Triton!")
    print(f"Model Name: {results.get_response()['model_name']}")
    print(f"Output tensor name: 'output0'")
    print(f"Output shape: {output_data.shape}")
    print("\nPhase 2 is officially complete. Congratulations!")


except Exception as e:
    print(f"\nERROR: An error occurred during inference: {e}")