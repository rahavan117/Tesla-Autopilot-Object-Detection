# Tesla Autopilot Clone - Object Detection
# YOLOv8n + KITTI Dataset

# Install Ultralytics
!pip install ultralytics -q

# Import Libraries
%env WANDB_DISABLED=True

from ultralytics import YOLO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import shutil
from PIL import Image

# Dataset Paths
base_dir = Path('/kaggle/input/kitti-dataset')

img_path = base_dir / 'data_object_image_2' / 'training' / 'image_2'

label_path = Path('/kaggle/input/kitti-dataset-yolo-format/labels')

# Load Classes
with open('/kaggle/input/kitti-dataset-yolo-format/classes.json', 'r') as f:
    classes = json.load(f)

print("Classes:")
print(classes)

# Load Images and Labels
ims = sorted(list(img_path.glob('*')))
labels = sorted(list(label_path.glob('*')))

print("Images:", len(ims))
print("Labels:", len(labels))

# Pair Images and Labels
pairs = list(zip(ims, labels))

print("Total Pairs:", len(pairs))
print("First two pairs:")
print(pairs[:2])

# Train Test Split
train, test = train_test_split(
    pairs,
    test_size=0.1,
    shuffle=True,
    random_state=42
)

print("Training Images:", len(train))
print("Validation Images:", len(test))

# Create Train and Validation Folders
train_path = Path('/kaggle/working/train')
train_path.mkdir(exist_ok=True)

valid_path = Path('/kaggle/working/valid')
valid_path.mkdir(exist_ok=True)

# Copy Training Images and Labels
for t_img, t_lb in tqdm(train, desc="Copying Training Data"):
    im_path = train_path / t_img.name
    lb_path = train_path / t_lb.name
    shutil.copy(t_img, im_path)
    shutil.copy(t_lb, lb_path)

# Copy Validation Images and Labels
for t_img, t_lb in tqdm(test, desc="Copying Validation Data"):
    im_path = valid_path / t_img.name
    lb_path = valid_path / t_lb.name
    shutil.copy(t_img, im_path)
    shutil.copy(t_lb, lb_path)

# Create KITTI YAML File
yaml_file = 'names:\n'
yaml_file += '\n'.join(f'- {c}' for c in classes)
yaml_file += f'\nnc: {len(classes)}'
yaml_file += f'\ntrain: {str(train_path)}'
yaml_file += f'\nval: {str(valid_path)}'

with open('/kaggle/working/kitti.yaml', 'w') as f:
    f.write(yaml_file)

print("\nKITTI YAML:")
print(yaml_file)

# Load YOLOv8n Model
model = YOLO('yolov8n.pt')

# Train YOLOv8n Model
train_results = model.train(
    data='/kaggle/working/kitti.yaml',
    epochs=10,
    patience=3,
    mixup=0.1,
    project='/kaggle/working/runs/detect/yolov8n-kitti',
    name='train',
    device=0
)

# Validate Model
valid_results = model.val()

print("\nValidation Results:")
print(valid_results)

# Results Directory
results_dir = Path(
    '/kaggle/working/runs/detect/yolov8n-kitti/train'
)

# Display Training Results
results_image = results_dir / 'results.png'

if results_image.exists():
    plt.figure(figsize=(10, 20))
    plt.imshow(Image.open(results_image))
    plt.axis('off')
    plt.show()
else:
    print("results.png not found")

# Display Confusion Matrix
confusion_matrix = results_dir / 'confusion_matrix.png'

if confusion_matrix.exists():
    plt.figure(figsize=(10, 20))
    plt.imshow(Image.open(confusion_matrix))
    plt.axis('off')
    plt.show()
else:
    print("confusion_matrix.png not found")

# Select Random Test Images
random_indices = np.random.randint(
    0,
    len(test),
    20
)

test_images = [
    test[idx][0]
    for idx in random_indices
]

print("Selected Test Images:", len(test_images))

# Predict Objects
predictions = model.predict(
    test_images,
    save=True
)

# Prediction Folder
prediction_dir = Path(
    '/kaggle/working/runs/detect/predict'
)

preds = list(prediction_dir.glob('*'))

print("Prediction Images:", len(preds))

# Display Prediction Images
def plot_images(images):
    num_images = len(images)
    cols = 4
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows,cols,figsize=(20, 5 * rows))

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis('off')

    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

plot_images(preds)

# Save Best Model
best_path = Path('/kaggle/working/runs/detect/yolov8n-kitti/train/weights/best.pt')

if best_path.exists():
    shutil.copy(best_path,'/kaggle/working/best.pt')

    print("Best model saved:")
    print("/kaggle/working/best.pt")
else:
    print("best.pt not found")

# Create Results ZIP File
shutil.make_archive('/kaggle/working/results','zip','/kaggle/working/runs/detect')

print("Results ZIP saved:")
print("/kaggle/working/results.zip")

# Final Output
print("\nProject completed successfully!")
print("Best Model: /kaggle/working/best.pt")
print("Results ZIP: /kaggle/working/results.zip")
