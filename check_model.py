import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from pathlib import Path
import random

CLASS_NAMES = [
    "acne", "actinic_keratosis", "benign_tumors", "bullous", "candidiasis",
    "drug_eruption", "eczema", "infestations_bites", "lichen", "lupus",
    "moles", "psoriasis", "rosacea", "seborrh_keratoses", "skin_cancer",
    "sun_sunlight_damage", "tinea", "unknown_normal", "vascular_tumors",
    "vasculitis", "vitiligo", "warts"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 22)
checkpoint = torch.load(r"C:\Local Disk D\efficientnet isic\outputs\skin_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset_path = Path("dataset/train")

print("Testing model on training images:")
print("=" * 50)

correct = 0
total = 0

for folder in sorted(dataset_path.iterdir()):
    if folder.is_dir():
        images = list(folder.glob("*"))
        if images:
            img_path = random.choice(images)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item() * 100
            
            actual = folder.name
            predicted = CLASS_NAMES[pred_idx]
            status = "✅" if actual == predicted else "❌"
            
            if actual == predicted:
                correct += 1
            total += 1
            
            print(f"{status} {actual}: predicted {predicted} ({confidence:.1f}%)")

print("=" * 50)
print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")