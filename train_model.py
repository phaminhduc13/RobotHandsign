import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Fix Unicode for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

IMG_SIZE = 64
CLASSES = ['wave', 'fist', 'point']

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_data():
    X, y = [], []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = f'dataset/{class_name}'
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(f'{class_dir}/{fname}')
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(label_idx)
    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    return X, y

def main():
    print("Loading data...")
    X, y = load_data()
    if len(X) == 0:
        print("Khong co du lieu! Chay collect_data.py truoc.")
        return

    print(f"Total: {len(X)} images")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {sum(y==i)}")

    # Chuyen thanh tensor (N, C, H, W)
    X = X.transpose(0, 3, 1, 2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining...")
    for epoch in range(30):
        model.train()
        total_loss, correct = 0, 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == batch_y).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/30 - Loss: {total_loss:.4f} - Acc: {acc*100:.2f}%")

    # Evaluate
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = outputs.argmax(1)
            all_preds.extend(preds.numpy())
            all_labels.extend(batch_y.numpy())
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    print(f"\n=== Test Accuracy: {correct/total*100:.2f}% ===")
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Save
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/gesture_classifier.pth')
    print("\nModel saved to models/gesture_classifier.pth")

if __name__ == "__main__":
    main()