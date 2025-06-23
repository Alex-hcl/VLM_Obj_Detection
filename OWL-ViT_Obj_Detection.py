import os, time, torch, numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import OwlViTProcessor, OwlViTForObjectDetection

IMAGE_ROOT = "./images"
LABEL_ROOT = "./labels"
PROMPT = [["truck"]]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(DEVICE)

imgs = [f for f in os.listdir(IMAGE_ROOT) if f.lower().endswith((".jpg",".png"))]

iou_accum, corr, tot = [], 0, 0
t0 = time.time()

for fn in tqdm(imgs, desc="OWL-ViT detect"):
    img = Image.open(os.path.join(IMAGE_ROOT,fn)).convert("RGB")
    w,h = img.size
    inp = processor(text=PROMPT, images=img, return_tensors="pt").to(DEVICE)
    out = model(**inp)
    boxes, scores, labels = processor.post_process_grounded_object_detection(
        out, target_sizes=torch.tensor([[h,w]]), threshold=0.3, text_labels=PROMPT
    )[0].values()

    preds = []
    for b,s,l in zip(boxes, scores, labels):
        if PROMPT[0][l] == "truck":
            preds.append([int(b[0]),int(b[1]),int(b[2]),int(b[3])])

    lbls = []
    pth = os.path.join(LABEL_ROOT, fn.replace(".jpg",".txt"))
    if os.path.exists(pth):
        for ln in open(pth):
            c,*v = ln.strip().split()
            if int(c)!=2: continue
            cx,cy,w_,h_ = map(float,v)
            x1=int((cx-w_/2)*w); y1=int((cy-h_/2)*h)
            x2=int((cx+w_/2)*w); y2=int((cy+h_/2)*h)
            lbls.append([x1,y1,x2,y2])

    if not preds or not lbls: continue
    for gt in lbls:
        ious = [(max(0, min(gt[2],p[2])-max(gt[0],p[0])+1)*
                 max(0, min(gt[3],p[3])-max(gt[1],p[1])+1)) /
                float((gt[2]-gt[0]+1)*(gt[3]-gt[1]+1)+(p[2]-p[0]+1)*(p[3]-p[1]+1)-
                      (max(0, min(gt[2],p[2])-max(gt[0],p[0])+1)*max(0, min(gt[3],p[3])-max(gt[1],p[1])+1)))
                for p in preds]
        m = max(ious) if ious else 0
        iou_accum.append(m)
        if m>0.5: corr+=1
        tot+=1

FPS = len(imgs)/(time.time()-t0)
mIoU = np.mean(iou_accum) if iou_accum else 0
acc = corr/tot*100 if tot else 0

print(f"\n=== OWL-ViT (truck) Evaluation ===")
print(f"Mean IoU     : {mIoU:.6f}")
print(f"Acc (IoU>0.5): {acc:.1f}%")
print(f"FPS          : {FPS:.2f}")
print("==============================")
