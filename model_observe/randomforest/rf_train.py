import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys
import os

# 프로젝트 루트 경로 설정 (기존 코드와 동일)
current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, augmentdata, label
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("전처리 파일(preprocess.py 또는 feature_extractor.py)을 찾을 수 없습니다.")
    exit()

class RFClassifierWrapper:
    def __init__(self, randomstate=42):
        self.randomstate = randomstate
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            random_state=self.randomstate,
            class_weight='balanced',
            n_jobs=-1
        )

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

def visualize_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Random Forest Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.show()

    print("\n[Top 5 Important Features]")
    for i in range(5):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

if __name__ == "__main__":
    data_path = os.path.join(project_root, 'data')
    xorig, yorig = load(data_path)

    if len(xorig) == 0:
        print("데이터가 없습니다. 경로를 확인해주세요.")
        exit()

    xtrainorig, xtestorig, ytrainorig, ytest = train_test_split(
        xorig, yorig, test_size=0.2, stratify=yorig, random_state=42
    )

    naugment = 9
    xauglist, ytrainaug = augmentdata(xtrainorig, ytrainorig, n=naugment)

    print("\n특성 추출 중...")
    xtrain_features, feature_names = extractfeatures(xauglist)
    xtest_features, _ = extractfeatures(xtestorig)

    print(f"학습 데이터 형태: {xtrain_features.shape}")
    print(f"테스트 데이터 형태: {xtest_features.shape}")

    rf_wrapper = RFClassifierWrapper()
    print("\n모델 학습 중 (Random Forest)...")
    rf_wrapper.fit(xtrain_features, ytrainaug)

    save_path = 'randomforest/rf.joblib'
    joblib.dump(rf_wrapper.model, save_path)
    print(f"모델 저장 완료: {save_path}")

    ypred = rf_wrapper.predict(xtest_features)
    
    classnames = list(label.keys())
    
    print("\n" + "="*30)
    print(f"Test Set Evaluation ({len(xtest_features)} samples)")
    print(classification_report(ytest, ypred, target_names=classnames, zero_division=0))
    
    cm = confusion_matrix(ytest, ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classnames, yticklabels=classnames)
    plt.title('Random Forest Confusion Matrix')
    plt.show()

    visualize_feature_importance(rf_wrapper.model, feature_names)