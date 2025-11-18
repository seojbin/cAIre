import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys
import os

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)

sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("전처리파일(preprocess.py, feature_extractor.py) 없음")
    exit()

model = os.path.join(script_dir, 'svm.joblib') # 불러올 모델 파일명 변경
newdata = os.path.join(project_root, 'newdata')

classnames = list(label.keys())

try:
    model = joblib.load(model)
except IOError:
    print(f"오류: {model}에 모델 없음.")
    exit()

xnew, ytrue = load(newdata) # 증강 안 함

if len(xnew) == 0:
    print("로드안됨")
    exit()

xnewfeatures, _ = extractfeatures(xnew)

print(f"총 {len(xnewfeatures)}개 새 데이터 추론")

ypred = model.predict(xnewfeatures)

print("추론 결과")
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