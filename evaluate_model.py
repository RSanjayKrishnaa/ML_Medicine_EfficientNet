import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
CLASS_NAMES = [
    "acne", "actinic_keratosis", "benign_tumors", "bullous", "candidiasis",
    "drug_eruption", "eczema", "infestations_bites", "lichen", "lupus",
    "moles", "psoriasis", "rosacea", "seborrh_keratoses", "skin_cancer",
    "sun_sunlight_damage", "tinea", "unknown_normal", "vascular_tumors",
    "vasculitis", "vitiligo", "warts"
]
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 16
MODEL_PATH = r"C:\Local Disk D\efficientnet isic\outputs\skin_model.pth"
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    print("Model loaded successfully!\n")
    
    # Load validation dataset
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder("dataset/val", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Validation samples: {len(val_dataset)}\n")
    print("=" * 60)
    print("EVALUATING MODEL...")
    print("=" * 60)
    
    # Collect predictions and labels
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(val_loader)
    
    # ROC-AUC (one-vs-rest for multiclass)
    try:
        roc_auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        roc_auc_weighted = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        roc_auc_macro = "N/A"
        roc_auc_weighted = "N/A"
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"1. Accuracy:           {accuracy * 100:.2f}%")
    print(f"2. F1 Score (Macro):   {f1_macro:.4f}")
    print(f"3. F1 Score (Weighted):{f1_weighted:.4f}")
    print(f"4. Average Loss:       {avg_loss:.4f}")
    print(f"5. ROC-AUC (Macro):    {roc_auc_macro if isinstance(roc_auc_macro, str) else f'{roc_auc_macro:.4f}'}")
    print(f"6. ROC-AUC (Weighted): {roc_auc_weighted if isinstance(roc_auc_weighted, str) else f'{roc_auc_weighted:.4f}'}")
    print("=" * 60)
    
    # Save detailed classification report
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    print("\nDetailed Classification Report:")
    print(report)
    
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write("EFFICIENTNET MODEL EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"F1 Score (Macro): {f1_macro:.4f}\n")
        f.write(f"F1 Score (Weighted): {f1_weighted:.4f}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"ROC-AUC (Macro): {roc_auc_macro if isinstance(roc_auc_macro, str) else f'{roc_auc_macro:.4f}'}\n")
        f.write(f"ROC-AUC (Weighted): {roc_auc_weighted if isinstance(roc_auc_weighted, str) else f'{roc_auc_weighted:.4f}'}\n\n")
        f.write(report)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - EfficientNet Model', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to {OUTPUT_DIR / 'confusion_matrix.png'}")
    
    # 2. ROC Curves (for each class)
    plt.figure(figsize=(12, 10))
    
    # Plot ROC curve for each class
    for i in range(min(10, NUM_CLASSES)):  # Plot first 10 classes to avoid clutter
        fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
        plt.plot(fpr, tpr, label=f'{CLASS_NAMES[i]}', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - EfficientNet Model (First 10 Classes)', fontsize=14, pad=15)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=300, bbox_inches='tight')
    print(f"✅ ROC curves saved to {OUTPUT_DIR / 'roc_curves.png'}")
    
    # 3. Per-class accuracy bar chart
    per_class_correct = np.zeros(NUM_CLASSES)
    per_class_total = np.zeros(NUM_CLASSES)
    
    for true, pred in zip(all_labels, all_preds):
        per_class_total[true] += 1
        if true == pred:
            per_class_correct[true] += 1
    
    per_class_accuracy = (per_class_correct / per_class_total) * 100
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(NUM_CLASSES), per_class_accuracy, color='steelblue', edgecolor='black')
    plt.xlabel('Disease Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy - EfficientNet Model', fontsize=14, pad=15)
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_class_accuracy.png", dpi=300, bbox_inches='tight')
    print(f"✅ Per-class accuracy chart saved to {OUTPUT_DIR / 'per_class_accuracy.png'}")
    
    print("\n" + "=" * 60)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'loss': avg_loss,
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted
    }

if __name__ == "__main__":
    metrics = evaluate_model()