from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import io
import logging

# --------------------------------------
# Logging
# --------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skin-app")

app = FastAPI()

# --------------------------------------
# Static + Template Setup
# --------------------------------------
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "templates" / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# --------------------------------------
# Load CSV Data Safely
# --------------------------------------
DATA_DIR = BASE_DIR / "data"
csv_files = {
    "description": "description.csv",
    "diets": "diets.csv",
    "medications": "medications.csv",
    "precautions": "precautions.csv",
    "symptoms": "symptoms.csv",
    "severity": "SymptomSeverity.csv",
    "training": "Training.csv",
    "workouts": "workout_df.csv"
}

csv_data = {}
for key, file in csv_files.items():
    try:
        df = pd.read_csv(DATA_DIR / file)
        
        # Normalize disease column if it exists
        if "disease" in df.columns:
            df["disease"] = df["disease"].astype(str).str.lower().str.strip()

        csv_data[key] = df
        logger.info(f"Loaded {file}")

    except Exception as e:
        logger.error(f"Failed to load {file}: {e}")
        csv_data[key] = pd.DataFrame()  # empty df fallback

# --------------------------------------
# Device setup
# --------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------
# Load Model
# --------------------------------------
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# Define all 22–23 classes
CLASS_NAMES = [
    "acne", "actinic_keratosis", "benign_tumors", "bullous", "candidiasis",
    "drug_eruption", "eczema", "infestations_bites", "lichen", "lupus",
    "moles", "psoriasis", "rosacea", "seborrh_keratoses", "skin_cancer",
    "sun_sunlight_damage", "tinea", "unknown_normal", "vascular_tumors",
    "vasculitis", "vitiligo", "warts"
]

# Replace classifier to match the number of classes
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))

# Load checkpoint safely
MODEL_PATH = r"C:\Local Disk D\efficientnet isic\outputs\skin_model.pth"
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

# Handle old checkpoints with different classifier size
state_dict = checkpoint.get("model_state", checkpoint)
model_state = {}
for k, v in state_dict.items():
    if k in model.state_dict() and v.size() == model.state_dict()[k].size():
        model_state[k] = v
    else:
        logger.warning(f"Skipping loading layer {k} due to size mismatch")

model.load_state_dict(model_state, strict=False)
model.eval()
model.to(device)
logger.info(f"Model loaded from {MODEL_PATH}")

# --------------------------------------
# Image transform
# --------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --------------------------------------
# Recommendation helper
# --------------------------------------
def get_recommendations(disease: str):

    # Normalize disease name for CSV matching
    disease = (
        disease.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .strip()
    )


    def safe_list(df, col):
        if df.empty or col not in df.columns or "disease" not in df.columns:
            return []
        filtered = df[df["disease"] == disease]
        return filtered[col].dropna().astype(str).tolist()[:5]

    return {
        "description": safe_list(csv_data["description"], "description"),
        "diets": safe_list(csv_data["diets"], "diet"),
        "medications": safe_list(csv_data["medications"], "medicine"),
        "precautions": safe_list(csv_data["precautions"], "precaution"),
        "symptoms": safe_list(csv_data["symptoms"], "symptom"),
        "severity": safe_list(csv_data["severity"], "severity"),
        "training": safe_list(csv_data["training"], "training"),
        "workouts": safe_list(csv_data["workouts"], "workout"),
    }

# --------------------------------------
# Routes
# --------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(image: UploadFile = File(...), prompt: str = Form("")):
    # Load image bytes
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_disease = CLASS_NAMES[predicted_idx.item()]

    # Load CSV-based recommendations
    rec = get_recommendations(predicted_disease)

    result = {
        "predicted_disease": predicted_disease,
        "description": rec["description"],
        "precautions": rec["precautions"],
        "medications": rec["medications"],
        "workouts": rec["workouts"],
        "diets": rec["diets"],
        "symptoms": rec["symptoms"],
        "severity": rec["severity"],
        "training": rec["training"],
        "prompt": prompt
    }

    return JSONResponse(content=result)
