# Multi Object Detection and Tracking

This project implements a multi-object detection and tracking system using YOLOv5 for object detection and DeepSORT for tracking. The system is capable of detecting multiple objects in video frames and tracking their movements across frames.

## Features
- Object detection using YOLOv8
- Multi-object tracking using DeepSORT
- Real-time FPS measurement and overlay on video frames

## Setup and Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/karpenet/DeepSORT-Multi-Object-Tracking
    cd multi-object-detection-tracking
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the YOLOv8 model:**
    - Download the YOLOv8 model and place it in the `models` directory. You can use the provided script to convert the model to TensorRT format:
    ```sh
    python:models/utils/compress_model.py
    ```

4. **Prepare the dataset:**
    - Place your dataset in the `dataset` directory. Ensure the images are in the correct format and directory structure.

## Running the Inference

1. **Run the inference script:**
    ```sh
    python inference.py
    ```

    This script will process the images in the dataset and perform object detection and tracking. The results will be saved in the `output` directory.


## Results

The results of the object detection and tracking will be saved as images or videos in the `output` directory. You can visualize the results using any image or video viewer.

![Result](output_deepsort.gif)

## Logging

Logs are saved in the `log.txt` file. You can check this file for detailed information about the processing steps and any errors encountered.

