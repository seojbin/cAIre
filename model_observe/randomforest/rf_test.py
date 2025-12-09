import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("전처리 파일을 찾을 수 없습니다.")
    exit()


if __name__ == "__main__":
    model_path = os.path.join(script_dir, 'rf.joblib')
    newdata_path = os.path.join(project_root, 'newdata')
    classnames = list(label.keys())

    try:
        model = joblib.load(model_path)
        print(f"모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        exit()

    xnew, ytrue = load(newdata_path)
    if len(xnew) == 0: 
        print("테스트할 데이터가 없습니다.")
        exit()

    print("\n특성 추출 중...")
    xnew_features, feature_names = extractfeatures(xnew)

    print(f"총 {len(xnew_features)}개 데이터 추론 시작")
    ypred = model.predict(xnew_features)

    predicted_labels = [classnames[p] for p in ypred]
    true_labels = [classnames[t] for t in ytrue]

    print("\n[추론 결과 샘플]")
    for i in range(min(10, len(ypred))): # 처음 10개만 출력
        print(f"Sample {i + 1}: 예측={predicted_labels[i]}, 실제={true_labels[i]}")

    print("\n" + "=" * 30)
    print(classification_report(ytrue, ypred, target_names=classnames, zero_division=0))

    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classnames, yticklabels=classnames)
    plt.title('Test Set Confusion Matrix (Random Forest)')
    plt.show()