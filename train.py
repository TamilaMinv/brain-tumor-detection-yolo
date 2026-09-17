"""
Brain Tumor Detection — YOLOv8n / YOLOv10n
Обучение модели детекции опухолей головного мозга (glioma, meningioma, pituitary)
на MRI-снимках с использованием фреймворка Ultralytics.

Датасет: BrainTumor (train: 2144, val: 612, test: 308 изображений)
"""

import os
import warnings
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")

print("CUDA доступна:", torch.cuda.is_available())
print("Устройство:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ---------------------------------------------------------------------------
# 1. Обучение модели
# ---------------------------------------------------------------------------

T_Model = YOLO("yolov8n.pt")
yaml_file_path = "BrainTumorYolov8/data.yaml"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"train_{timestamp}"
print(f"Run name: {run_name}")

# optimizer: 'auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'
results = T_Model.train(
    data=yaml_file_path,
    epochs=30,
    patience=15,
    batch=4,
    optimizer="SGD",
    name="yolo8n1_brain_run",
    lr0=0.01,
)

post_training_files_path = str(results.save_dir)
print(f"Модель сохранена в: {post_training_files_path}")

# ---------------------------------------------------------------------------
# 2. Валидация лучшей модели
# ---------------------------------------------------------------------------

best_model_path = os.path.join(post_training_files_path, "weights/best.pt")
best_model = YOLO(best_model_path)

metrics = best_model.val(split="val")
metrics_dict = metrics.results_dict
metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index", columns=["Metric Value"])
print(metrics_df)

# ---------------------------------------------------------------------------
# 3. Экспорт и визуализация кривых обучения (Cls Loss, mAP)
# ---------------------------------------------------------------------------

csv_path = os.path.join(post_training_files_path, "results.csv")
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

output_csv = os.path.join(post_training_files_path, "losses_export.csv")
metric_cols = (
    ["epoch"]
    + [c for c in df.columns if "cls" in c.lower() and "loss" in c.lower()]
    + [c for c in df.columns if "map" in c.lower()]
)
df[metric_cols].to_csv(output_csv, index=False)
print(f"Cls Loss + mAP метрики сохранены в: {output_csv}")

train_cls_cols = [c for c in df.columns if "train" in c.lower() and "cls" in c.lower() and "loss" in c.lower()]
val_cls_cols = [c for c in df.columns if "val" in c.lower() and "cls" in c.lower() and "loss" in c.lower()]

fig, ax = plt.subplots(figsize=(8, 5))
if train_cls_cols:
    ax.plot(df["epoch"], df[train_cls_cols[0]], label="Train Cls Loss", color="steelblue")
if val_cls_cols:
    ax.plot(df["epoch"], df[val_cls_cols[0]], label="Val Cls Loss", color="tomato")
ax.set_title("Cls Loss: Train vs Val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cls Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(post_training_files_path, "cls_loss_curve.png"), dpi=120, bbox_inches="tight")
plt.show()

map50_cols = [c for c in df.columns if "map50" in c.lower() and "map50-95" not in c.lower()]
map5095_cols = [c for c in df.columns if "map50-95" in c.lower()]

fig, ax = plt.subplots(figsize=(8, 5))
if map50_cols:
    ax.plot(df["epoch"], df[map50_cols[0]], label="Val mAP@50 — точность при пороге IoU=0.50", color="mediumseagreen")
if map5095_cols:
    ax.plot(
        df["epoch"],
        df[map5095_cols[0]],
        label="Val mAP@50-95 — средняя точность по порогам IoU 0.50–0.95",
        color="darkorange",
    )
ax.set_title("mAP метрики")
ax.set_xlabel("Epoch")
ax.set_ylabel("mAP")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(post_training_files_path, "map_combined.png"), dpi=120, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 4. Визуализация служебных графиков (F1/P/R/PR-кривые, confusion matrix)
# ---------------------------------------------------------------------------

def display_images_in_frame_with_background(post_training_files_path, image_files, background_color="lightgray"):
    num_images = len(image_files)
    cols = 3
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), dpi=120)
    fig.patch.set_facecolor(background_color)
    axes = axes.flatten()

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(post_training_files_path, image_file)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(image_file, fontsize=10, color="black")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


image_files = [
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxPR_curve.png",
    "confusion_matrix_normalized.png",
]
display_images_in_frame_with_background(post_training_files_path, image_files, background_color="lightblue")


# ---------------------------------------------------------------------------
# 5. Инференс на тестовой выборке и визуализация предсказаний
# ---------------------------------------------------------------------------

def normalize_image(image):
    return image / 255.0


def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)


dataset_path = r"BrainTumorYolov8"
valid_images_path = os.path.join(dataset_path, "test", "images")

image_files = [file for file in os.listdir(valid_images_path) if file.endswith(".jpg")]

if len(image_files) > 0:
    num_images = len(image_files)
    step_size = max(1, num_images // 16)
    selected_images = [image_files[i] for i in range(0, num_images, step_size)]

    fig, axes = plt.subplots(4, 4, figsize=(20, 21))
    fig.suptitle("TEST Set Inferences", fontsize=24)

    for i, ax in enumerate(axes.flatten()):
        if i < len(selected_images):
            image_path = os.path.join(valid_images_path, selected_images[i])
            image = cv2.imread(image_path)

            if image is not None:
                resized_image = resize_image(image, size=(640, 640))
                normalized_image = normalize_image(resized_image)
                normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)

                results = best_model.predict(source=normalized_image_uint8, imgsz=640, conf=0.5)
                annotated_image = results[0].plot(line_width=1)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                ax.imshow(annotated_image_rgb)
            else:
                print(f"Failed to load image {image_path}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
