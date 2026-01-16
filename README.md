# YoloAutoLabel
A desktop GUI tool that uses YOLO models to auto-generate bounding box annotations for image datasets—preview, filter classes, and batch-export in YOLO format.

# YOLO Auto-Annotation Tool

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-blue)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)

A powerful and user-friendly desktop application built with Python and PyQt5 that leverages YOLO (You Only Look Once) models to automatically generate bounding box annotations for your image datasets.

## 📌 Overview

Manually annotating thousands of images is a tedious and time-consuming task. This tool aims to solve that problem by providing a simple GUI to:
1.  **Preview** detections from your custom YOLO model on your images.
2.  **Select specific classes** you want to annotate, ignoring others.
3.  **Automatically generate** YOLO-format `.txt` label files for your entire dataset.
4.  **Generate a `classes.txt`** file that maps the new class IDs in your labels to their human-readable names.

It's perfect for bootstrapping your object detection projects or quickly labeling large datasets.

## ✨ Features

*   **Intuitive GUI**: Easy-to-use interface built with PyQt5.
*   **Model Agnostic**: Works with any YOLOv5/v8 `.pt` model file.
*   **Class Filtering**: Choose exactly which object classes to annotate.
*   **Real-time Preview**: See how your model performs on your images before full annotation.
*   **Confidence Threshold**: Adjust the detection confidence to fine-tune results.
*   **Batch Processing**: Annotate entire directories of images with a single click.
*   **Standard Output**: Generates labels in the standard YOLO format (`class_id x_center y_center width height`).

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/yolo-auto-annotation-tool.git
    cd yolo-auto-annotation-tool
    ```

2.  **Install dependencies**:
    It's highly recommended to use a virtual environment.
    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install required packages
    pip install -r requirements.txt
    ```
    *(Note: You'll need to create a `requirements.txt` file with the following content)*:
    ```
    opencv-python
    ultralytics
    torch
    torchvision
    PyQt5
    ```

3.  **Prepare your model**:
    Place your trained YOLO `.pt` model file into the `models/` directory.

## 🛠️ Usage

1.  **Run the application**:
    ```bash
    python main.py
    ```

2.  **In the GUI**:
    *   **Select Model**: Choose your `.pt` model from the dropdown.
    *   **Select Classes**: Check the boxes for the object classes you want to annotate.
    *   **Set Directories**: Choose your input image directory and the output label directory.
    *   **Adjust Confidence**: Use the slider to set the detection confidence threshold.
    *   **Preview**: Click "Load & Preview Images" to see how the model detects objects on your images.
    *   **Annotate**: Once satisfied, click "Start Auto-Annotation" to process all images.

The tool will create a `.txt` label file for each image in your specified output directory and generate a `classes.txt` file listing the annotated classes in order.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for bug fixes, new features, or documentation improvements.

---
**Author**: YouLuoYuan TuBoShu，My Web Site：www.youluoyuan.com

![Preview](https://raw.githubusercontent.com/PainLxrd/YoloAutoLabel/refs/heads/main/preview.png)
