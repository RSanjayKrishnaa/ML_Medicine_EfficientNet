from torchvision import datasets

train_dataset = datasets.ImageFolder("dataset/train")

print("Correct class order from training:")
print(train_dataset.classes)

print("\nClass to index mapping:")
print(train_dataset.class_to_idx)