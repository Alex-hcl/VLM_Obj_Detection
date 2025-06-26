import os, json
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from groundingdino.models import build_groundingdino
from groundingdino.util.slconfig import SLConfig
from groundingdino.engine import train_one_epoch
from groundingdino.util.utils import clean_state_dict

class GroundingDinoDataset(Dataset):
    def __init__(self, ann_file, img_root, transform=None):
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        self.root = img_root
        self.transform = transform or T.Compose([
            T.Resize((800, 800)),
            T.ToTensor()
        ])

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = os.path.join(self.root, item['file_name'])
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)

        boxes = torch.tensor(item['boxes'], dtype=torch.float32)
        phrases = item['phrases']
        return {"image": img_tensor, "boxes": boxes, "phrases": phrases}

def build_model(config_path, checkpoint_path):
    args = SLConfig.fromfile(config_path)
    args.device = "cuda"
    model = build_groundingdino(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    return model

train_json = "data/1.txt"
img_folder = "data/images"
config = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
weights = "weights/groundingdino_swint_ogc.pth"

dataset = GroundingDinoDataset(train_json, img_folder)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: x)
model = build_model(config, weights).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(10):
    train_one_epoch(model, optimizer, loader, device="cuda", epoch=epoch)
    torch.save(model.state_dict(), f"checkpoints/groundingdino_epoch{epoch}.pth")
