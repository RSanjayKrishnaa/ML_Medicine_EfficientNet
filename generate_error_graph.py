import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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

def generate_error_rate_graph():
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
    
    print("Calculating per-class error rates...\n")
    
    # Collect predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate per-class error rates
    error_rates = []
    class_counts = []
    
    for i in range(NUM_CLASSES):
        # Get samples for this class
        class_mask = (all_labels == i)
        class_total = class_mask.sum()
        
        if class_total > 0:
            # Get incorrect predictions for this class
            class_preds = all_preds[class_mask]
            class_true = all_labels[class_mask]
            errors = (class_preds != class_true).sum()
            error_rate = (errors / class_total) * 100
        else:
            error_rate = 0
        
        error_rates.append(error_rate)
        class_counts.append(class_total)
    
    # Create the graph
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x_pos = np.arange(NUM_CLASSES)
    colors = ['#d62728' if er > 30 else '#ff7f0e' if er > 20 else '#2ca02c' for er in error_rates]
    
    bars = ax.bar(x_pos, error_rates, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Disease Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Error Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Error Rate - EfficientNet Model\n(Lower is Better)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, max(error_rates) + 10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, er, count) in enumerate(zip(bars, error_rates, class_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{er:.1f}%\n(n={count})',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', label='Good (≤20% error)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Moderate (20-30% error)'),
        Patch(facecolor='#d62728', edgecolor='black', label='High (>30% error)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add overall stats box
    overall_error = 100 - (all_preds == all_labels).mean() * 100
    textstr = f'Overall Error Rate: {overall_error:.2f}%\nOverall Accuracy: {100-overall_error:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the graph
    output_path = OUTPUT_DIR / "error_rate_graph.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Error rate graph saved to: {output_path}\n")
    
    # Print summary
    print("=" * 70)
    print("ERROR RATE SUMMARY")
    print("=" * 70)
    print(f"{'Disease':<25} {'Error Rate':<15} {'Samples':<10}")
    print("-" * 70)
    
    # Sort by error rate (highest first)
    sorted_indices = np.argsort(error_rates)[::-1]
    
    for idx in sorted_indices:
        status = "⚠️ " if error_rates[idx] > 30 else "✓ " if error_rates[idx] <= 20 else "○ "
        print(f"{status}{CLASS_NAMES[idx]:<23} {error_rates[idx]:>6.2f}%          {class_counts[idx]:<10}")
    
    print("-" * 70)
    print(f"Overall Error Rate: {overall_error:.2f}%")
    print(f"Overall Accuracy: {100-overall_error:.2f}%")
    print("=" * 70)
    
    plt.show()

if __name__ == "__main__":
    generate_error_rate_graph()