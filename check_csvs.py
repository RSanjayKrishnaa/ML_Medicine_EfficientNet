import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# Your correct class names from training
CLASS_NAMES = [
    "acne", "actinic_keratosis", "benign_tumors", "bullous", "candidiasis",
    "drug_eruption", "eczema", "infestations_bites", "lichen", "lupus",
    "moles", "psoriasis", "rosacea", "seborrh_keratoses", "skin_cancer",
    "sun_sunlight_damage", "tinea", "unknown_normal", "vascular_tumors",
    "vasculitis", "vitiligo", "warts"
]

csv_files = ["description.csv", "medications.csv", "workout_df.csv", "diets.csv", "precautions.csv"]

for file in csv_files:
    try:
        df = pd.read_csv(DATA_DIR / file)
        if "disease" in df.columns:
            diseases_in_csv = df["disease"].str.lower().str.strip().unique().tolist()
            print(f"\n=== {file} ===")
            print(f"Diseases in CSV: {diseases_in_csv}")
            
            # Check for mismatches
            missing = [c for c in CLASS_NAMES if c not in diseases_in_csv]
            extra = [d for d in diseases_in_csv if d not in CLASS_NAMES]
            
            if missing:
                print(f"❌ MISSING from CSV: {missing}")
            if extra:
                print(f"⚠️  EXTRA in CSV (not in model): {extra}")
            if not missing and not extra:
                print("✅ Perfect match!")
    except Exception as e:
        print(f"\n=== {file} === ERROR: {e}")