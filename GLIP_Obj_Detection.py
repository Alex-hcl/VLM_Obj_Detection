import os, time
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from glip.models import build_model
from glip.processor import GLIPProcessor

img_root = "./images"
label_root = "./labels"
config_path = "./GLIP/configs/glip_Swin_T_O365_GoldG.yaml"
ckpt_path = "./glip_tiny_model.pth"

text_prompt = "truck"
device = torch.device("cuda:0")

model = build_model(config_path, ckpt_path, device)
processor = GLIPProcessor.from_pretrained("GLIP")

image_files = [f for f in os.listdir(img_root) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"Total images loaded: {len(image_files)}")

iou_list, det_correct, det_total = [], 0, 0
t0 = time.time()

for img_name in tqdm(image_files, desc="GLIP processing"):
    img_path = os.path.join(img_root, img_name)
    lbl_path = os.path.join(label_root, img_name.replace(".jpg", ".txt"))
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    
    logits, boxes = out['logits'], out['boxes']
    keep = logits.squeeze() > 0.5
    pred_boxes = []

    if keep.sum() > 0:
        box_tensor = boxes[keep]
        box_tensor[:, 0::2] *= w
        box_tensor[:, 1::2] *= h
        pred_boxes = box_tensor.cpu().numpy().astype(int).tolist()

    gt_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls = int(parts[0])
                if cls != 2: continue
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                gt_boxes.append([x1, y1, x2, y2])

    if not gt_boxes or not pred_boxes: continue

    for g in gt_boxes:
        ious = [ (lambda a,b: (max(0, min(a[2],b[2])-max(a[0],b[0])+1) * max(0, min(a[3],b[3])-max(a[1],b[1])+1)) / float( (a[2]-a[0]+1)*(a[3]-a[1]+1) + (b[2]-b[0]+1)*(b[3]-b[1]+1) - (max(0, min(a[2],b[2])-max(a[0],b[0])+1) * max(0, min(a[3],b[3])-max(a[1],b[1])+1)) ))(g,p) for p in pred_boxes ]
        max_iou = max(ious) if ious else 0
        iou_list.append(max_iou)
        if max_iou > 0.5: det_correct += 1
        det_total += 1

t1 = time.time()
fps = len(image_files) / (t1 - t0)
mIoU = np.mean(iou_list) if iou_list else 0
acc = (det_correct / det_total * 100) if det_total else 0

print("\n==================== GLIP Detection Result ====================")
print(f"Average IoU (Truck)      : {mIoU:.6f}")
print(f"Accuracy (IoU > 0.5)     : {acc:.1f}%")
print(f"Avg Inference Speed (FPS): {fps:.2f}")
print("===============================================================\n")
