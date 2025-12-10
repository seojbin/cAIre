import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys
import os
from svm_train_knn import Hybrid_SVM_KNN
from tensorflow.keras.preprocessing.sequence import pad_sequences


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

model_filename = 'svm_knn.joblib'
model_path = os.path.join(script_dir, model_filename)
newdata = os.path.join(project_root, 'newdata')

classnames = list(label.keys())

try:
    model = joblib.load(model_path)
except IOError:
    print(f"오류: {model_path}에 모델 없음.")
    exit()
except ImportError:
    print(f"오류: Hybrid_SVM_KNN 클래스를 찾을 수 없습니다.")
    exit()

xnew, ytrue = load(newdata)

if len(xnew) == 0:
    print("로드안됨")
    exit()

xnewfeatures, _ = extractfeatures(xnew)

print(f"총 {len(xnewfeatures)}개 새 데이터 추론")

ypred = model.predict(xnewfeatures, xnew)

print("추론 결과")
predicted_labels = [classnames[p] for p in ypred]
true_labels = [classnames[t] for t in ytrue]

try:
    dtw_train_internal_labels = model.model2._y # m2로 변경 (이전 model4)
    dtw_original_labels_map = model.model2.classes_ 
    dtw_train_classnames = [classnames[ dtw_original_labels_map[l] ] for l in dtw_train_internal_labels]
except Exception as e:
    print(f"DTW 설명 라벨 로드 중 오류: {e}")
    dtw_train_classnames = []

for i in range(len(predicted_labels)):
    print(f"샘플 {i+1}: 예측={predicted_labels[i]}, 실제={true_labels[i]}")

    if predicted_labels[i] in ['diagonal_left', 'diagonal_right']:
        try:
            sample_orig_3d = [xnew[i]]
            sample_padded = pad_sequences(sample_orig_3d, padding='post', dtype='float32', value=np.nan)
            distances, indices = model.model2.kneighbors(sample_padded)
            
            neighbor_indices = indices[0]
            neighbor_labels = [dtw_train_classnames[idx] for idx in neighbor_indices]
            
            print(f"  [DTW 근거 (k=3)]: {neighbor_labels}")
            
        except Exception as e:
            print(f"  [DTW 설명 중 오류]: {e}")

if len(ytrue) > 0:
    print(classification_report(ytrue, ypred, target_names=classnames, zero_division=0))
    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classnames, yticklabels=classnames)
    plt.title('Data Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()