# Tesla-Autopilot-Object-Detection
YOLOv8n based object detection using the KITTI dataset
# Tesla Autopilot Clone – Object Detection

## Project Overview

This project is a Tesla Autopilot-inspired object detection prototype developed using YOLOv8n and the KITTI Object Detection Dataset.

The model detects different objects from road images.

## Classes

- Car
- Pedestrian
- Van
- Cyclist
- Truck
- Misc
- Tram
- Person_sitting

## Technologies Used

- Python
- YOLOv8
- Ultralytics
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- tqdm
- Pillow
- Jupyter Notebook
- Kaggle

## Dataset

KITTI Object Detection Dataset

Total Images: 7481

Training Images: 6732

Validation Images: 749

The complete dataset is not included in this GitHub repository because of its large size.

## Model

YOLOv8n is used for object detection.

## Training Configuration

- Epochs: 10
- Patience: 3
- Mixup: 0.1
- Device: GPU

## Model Performance

| Metric | Score |
|---|---:|
| Precision | 0.817 |
| Recall | 0.601 |
| mAP@50 | 0.704 |
| mAP@50-95 | 0.444 |

## Features

- Road object detection
- Multiple object class detection
- YOLOv8n model training
- Model validation
- Confusion matrix
- Training result visualization
- Test image prediction
- Best model saving

## Project Structure

```text
Tesla-Autopilot-Clone-Object-Detection/
│
├── Tesla_Autopilot_Object_Detection.py
├── Tesla_Autopilot_Object_Detection.ipynb
├── README.md
├── requirements.txt
├── kitti.yaml
└── .gitignore
