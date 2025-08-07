# PreVision Stream: Real-Time Video Analytics Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A comprehensive, end-to-end computer vision pipeline that performs real-time object detection on video streams using a custom-trained YOLOv8 model, deployed on a production-grade inference server.

---

### 🎥 Live Demo

"Uploading SOON"

### ✨ Key Features

* **Real-Time Inference:** Processes video streams with low latency for immediate object detection.
* **Custom-Trained Model:** Utilizes a YOLOv8 model fine-tuned on a specific dataset for high accuracy.
* **Production-Grade Deployment:** The model is optimized with OpenVINO and served via NVIDIA Triton Inference Server in a Docker container.
* **Scalable Architecture:** The client-server architecture decouples the AI model from the application, allowing them to be scaled and updated independently.
* **Hardware-Agnostic Inference:** Demonstrates deployment on a CPU-based backend, showcasing adaptability.

---

### 🛠️ Tech Stack & Architecture

This project leverages a modern stack for AI model deployment and application development.

**Technology:**
* **Model:** Python, PyTorch, YOLOv8
* **Optimization:** Intel OpenVINO Toolkit
* **Deployment:** NVIDIA Triton Inference Server, Docker
* **Application:** Python, OpenCV, NumPy

**Architecture:**

*(Create a simple diagram using a free tool like `draw.io` or `Excalidraw` and embed the image here. Show the flow: Video -> OpenCV App -> Preprocessing -> Triton Server -> Post-processing -> Visualization)*

![Architecture Diagram](link_to_your_diagram.png)

---

### ⚙️ Setup & Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/prevision-stream.git](https://github.com/YourUsername/prevision-stream.git)
    cd prevision-stream
    ```

2.  **Set up a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You should create a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated virtual environment)*

---

### ▶️ Usage

1.  **Start the Triton Inference Server:**
    Make sure Docker Desktop is running. Navigate to the project root and run:
    ```bash
    # Ensure the path in the -v flag is the absolute path to your triton_model_repository
    docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v "$(pwd)/triton_model_repository":/models nvcr.io/nvidia/tritonserver:24.07-py3 tritonserver --model-repository=/models
    ```

2.  **Run the Main Application:**
    In a separate terminal, run the OpenCV application:
    ```bash
    python3 app/main_app.py
    ```

---

### 🧠 Challenges & Learnings

* **Challenge:** Encountered a `port is already allocated` error during deployment.
    * **Solution:** Learned to debug Docker environments by listing and stopping previously running containers using `docker ps` and `docker stop`.
* **Challenge:** Triton server failed to start due to a `venv` folder inside the model repository.
    * **Solution:** Understood the importance of a clean deployment directory. Triton treats every subdirectory as a model, so the repository must only contain valid models. This was fixed by restructuring the project directory.