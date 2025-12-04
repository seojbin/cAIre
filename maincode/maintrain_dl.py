import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label
    from postprocess.augment_dl import augment_data
    from postprocess.featurizer_dl import extract_features_dl
except ImportError as e:
    print(f"\n[Import Error] {e}")
    exit()

# --- Config ---
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUGMENT_FACTOR = 5 
TARGET_LEN = 100 # 시계열 길이 고정

# --- Model Definition ---
class GestureModel(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_size=64):
        super(GestureModel, self).__init__()
        # input_channels = 3 (x, y, z)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Channels) = (32, 100, 3)
        x = x.permute(0, 2, 1) # -> (Batch, Channels, Seq_Len) for CNN
        x = self.cnn(x)
        x = x.permute(0, 2, 1) # -> (Batch, New_Seq_Len, Channels) for LSTM
        _, (hn, _) = self.lstm(x)
        x = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        x = self.fc(x)
        return x

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    return history

def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(loader), correct / total

# --- Main ---
def main():
    print(f"Using device: {DEVICE}")
    data_path = os.path.join(project_root, 'data')
    
    try:
        x_orig, y_orig = load(data_path)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    if len(x_orig) == 0:
        print("데이터가 없습니다.")
        return

    # Split
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
        x_orig, y_orig, test_size=0.2, stratify=y_orig, random_state=42
    )
    
    # Augmentation
    print(f"Augmenting training data (x{AUGMENT_FACTOR})...")
    x_train_aug = []
    y_train_aug = []
    
    for x, y in zip(x_train_raw, y_train_raw):
        # 원본 추가
        x_train_aug.append(x)
        y_train_aug.append(y)
        
        # 증강 데이터 추가 (리스트 반환)
        aug_list = augment_data(x, n=AUGMENT_FACTOR)
        x_train_aug.extend(aug_list)
        y_train_aug.extend([y] * len(aug_list))

    print(f"Augmented Train Size: {len(x_train_aug)}")
    
    # Feature Extraction (Sequence Conversion)
    print("Extracting features (Resampling & Normalizing)...")
    X_train = extract_features_dl(x_train_aug, target_len=TARGET_LEN)
    X_test = extract_features_dl(x_test_raw, target_len=TARGET_LEN)

    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    # 예상 Shape: (N, 100, 3)
    
    # Tensor Conversion
    y_train_tensor = torch.LongTensor(np.array(y_train_aug))
    y_test_tensor = torch.LongTensor(np.array(y_test_raw))
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model Init
    input_channels = 3 # (x, y, z)
    num_classes = len(label)
    
    model = GestureModel(input_channels, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    
    # Save
    torch.save(model.state_dict(), 'dl_model.pth')
    print("Model saved.")

    # Visualization (Loss)
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()