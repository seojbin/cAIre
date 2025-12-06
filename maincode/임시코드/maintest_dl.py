import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- 경로 설정 ---
current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label
    # [수정] 패키지 경로를 postprocess로 변경
    from postprocess.featurizer_dl import extract_features_dl
except ImportError as e:
    print(f"\n[Import Error] 모듈을 찾을 수 없습니다: {e}")
    print("해결: 'featurizer_dl.py'가 'postprocess' 폴더에 있는지 확인하세요.")
    exit()

# --- Config ---
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'dl_model.pth'
TARGET_LEN = 100  # [중요] 학습 코드와 동일한 길이 사용

# --- Model Definition (maintrain_dl.py와 100% 동일해야 함) ---
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
        # x shape: (Batch, Seq, Feat) -> (Batch, Feat, Seq) for CNN
        x = x.permute(0, 2, 1) 
        x = self.cnn(x)
        x = x.permute(0, 2, 1) # -> (Batch, Seq, Feat) for LSTM
        _, (hn, _) = self.lstm(x)
        x = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1) # Bi-directional
        x = self.fc(x)
        return x

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load New Data
    newdata_path = os.path.join(project_root, 'newdata')
    print(f"Loading test data from: {newdata_path}")
    
    try:
        x_new, y_new = load(newdata_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    if len(x_new) == 0:
        print("No data found in 'newdata' folder.")
        return

    # 2. Feature Extraction (Resampling & Normalization)
    print("Extracting features...")
    
    # [수정] 수동 패딩 로직 제거 -> extract_features_dl 내부에서 resample 수행
    # 반환 형태: (N, 100, 3)
    X_test = extract_features_dl(x_new, target_len=TARGET_LEN)
    
    print(f"Test Data Shape: {X_test.shape}") # (N, 100, 3) 확인

    # Tensor conversion
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_new)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Please train the model first.")
        return

    # 파라미터 설정 (학습 코드와 일치)
    input_channels = 3 
    num_classes = len(label)
    
    model = GestureModel(input_channels, num_classes).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except RuntimeError as e:
        print(f"\n[Model Load Error] 모델 구조가 일치하지 않습니다.\n{e}")
        print("Tip: maintrain_dl.py를 실행하여 모델을 다시 학습시키세요.")
        return

    model.eval()
    
    # 4. Predict
    print("Running prediction...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 5. Report
    # 라벨 이름 정렬 (0, 1, 2, 3, 4 순서)
    class_names = [k for k, v in sorted(label.items(), key=lambda item: item[1])]
    
    print("\n" + "="*30)
    print(f"Test Set Evaluation ({len(x_new)} samples)")
    
    # 개별 샘플 결과 출력
    for i in range(len(all_preds)):
        pred_str = class_names[all_preds[i]]
        true_str = class_names[all_labels[i]]
        print(f"Sample {i+1}: Predict={pred_str}, True={true_str}")
        
    print("\n" + "="*30)
    # 분류 보고서
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # Confusion Matrix 시각화
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Deep Learning Test Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    main()