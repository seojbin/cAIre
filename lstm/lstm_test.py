import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import os

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)

sys.path.append(project_root)
try:
    from postprocess.preprocess import load, pad, label
except ImportError:
    print("전처리파일 없음")
    exit()

model = os.path.join(script_dir, 'lstm.keras') #학습파일
newdata = os.path.join(project_root, 'newdata')
n = len(label)
classnames = list(label.keys())

try:
    model = load_model(model)
except IOError:
    print(f"{model}에 모델 없음")
    exit()

# 데이터 로드
xlist, ytrue = load(newdata)

if len(xlist) == 0:
    print("로드안됨")
    exit()

#새로운 데이터를 패딩
xpadded = pad(xlist)

print(f"총 {len(xpadded)}개의 새 데이터 추론 시작...")

ypredprobs = model.predict(xpadded)
ypred = np.argmax(ypredprobs, axis=1) # 확률이 가장 높은 클래스 인덱스 추출

print("예측 / 실제 라벨 / 파일 경로(추정)")
print("-" * 40)

predicted_labels = [classnames[p] for p in ypred]
true_labels = [classnames[t] for t in ytrue]

for i in range(len(predicted_labels)):
    print(f"샘플 {i+1}: 예측={predicted_labels[i]}, 실제={true_labels[i]}")

if len(ytrue) > 0:
    print(classification_report(ytrue, ypred, target_names=classnames))

    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classnames, yticklabels=classnames)
    plt.title('Data Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
