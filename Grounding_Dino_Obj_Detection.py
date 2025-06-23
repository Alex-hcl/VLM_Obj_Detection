import os
import time
import torch
import numpy as np
import cv2
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict

image_dir = "./images"
label_dir = "./labels"
dino_config_path = "./GroundingDINO/config/GroundingDINO_SwinT_OGC.py"
dino_model_path = "./groundingdino_swint_ogc.pth"

prompt_text = "truck"
device = torch.device("cuda:0")

print("Loading Grounding DINO model...")
model = load_model(dino_config_path, dino_model_path).to(device).eval()

image_list = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"Found {len(image_list)} images in '{image_dir}'.")

ious, correct_detections, total_detections = [], 0, 0
start_time = time.time()

for image_name in tqdm(image_list, desc="Processing with Grounding DINO"):
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))

    image_source, image_tensor = load_image(image_path)
    h, w = image_source.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt_text,
        box_threshold=0.35,
        text_threshold=0.25
    )

    pred_boxes = []
    for box in boxes:
        x_min = int(box[0] * w)
        y_min = int(box[1] * h)
        x_max = int(box[2] * w)
        y_max = int(box[3] * h)
        pred_boxes.append([x_min, y_min, x_max, y_max])

    gt_boxes = load_gt_txt_label(label_path, w, h, target_class_id=2)

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        continue

    for gt_box in gt_boxes:
        best_iou = max([compute_iou(gt_box, pred_box) for pred_box in pred_boxes] or [0])
        ious.append(best_iou)
        if best_iou > 0.5:
            correct_detections += 1
        total_detections += 1

end_time = time.time()
elapsed_time = end_time - start_time
avg_iou = np.mean(ious) if ious else 0
accuracy = (correct_detections / total_detections * 100) if total_detections else 0
fps = len(image_list) / elapsed_time

print("\n==================== Grounding DINO Detection Summary ====================")
print(f"Average IoU (Truck): {avg_iou:.6f}")
print(f"Accuracy (IoU > 0.5): {accuracy:.1f}%")
print(f"Average Inference Speed: {fps:.2f} FPS")
print("==========================================================================\n")

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def load_gt_txt_label(txt_path, img_w, img_h, target_class_id):
    bboxes = []
    if not os.path.exists(txt_path):
        print(f"Warning: label file not found: {txt_path}")
        return bboxes
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            if class_id != target_class_id:
                continue
            cx, cy, w, h = map(float, parts[1:])
            x_min = int((cx - w / 2) * img_w)
            y_min = int((cy - h / 2) * img_h)
            x_max = int((cx + w / 2) * img_w)
            y_max = int((cy + h / 2) * img_h)
            bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes
