import openvino as ov
import os

# --- Configuration ---
# Make sure this matches the name of your uploaded file
onnx_model_path = "./best.onnx"
# This will be the directory where the optimized model is saved
output_dir = "openvino_model"

# --- Conversion ---
print("Converting the ONNX model to OpenVINO IR format...")

# Step 1: Create an OpenVINO Core object
core = ov.Core()

# Step 2: Convert the model. OpenVINO reads the ONNX model and creates an optimized in-memory representation.
ov_model = core.read_model(onnx_model_path)

# Step 3: Save the optimized model to disk. This creates the .xml and .bin files.
os.makedirs(output_dir, exist_ok=True)
ov.save_model(ov_model, os.path.join(output_dir, "model.xml"))

print(f"Conversion successful! Optimized model saved in '{output_dir}' directory.")
print("You should now see 'model.xml' and 'model.bin' inside the new folder.")