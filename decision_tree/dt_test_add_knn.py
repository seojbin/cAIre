import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys
import os
from sklearn.tree import DecisionTreeClassifier 
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesResampler
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

class HybridClassifier:
    def __init__(self,maxdepthmain=3,maxdepthsimple=3,maxdepthcomplex=3,maxneighbor=3,randomstate=42):
        self.randomstate = randomstate
        
        # 3개의 DT 모델 - 깊이는 3으로 일단 설정
        self.model1 = DecisionTreeClassifier(max_depth=maxdepthmain, random_state=self.randomstate)
        self.model2 = DecisionTreeClassifier(max_depth=maxdepthsimple, random_state=self.randomstate)
        self.model3 = DecisionTreeClassifier(max_depth=maxdepthcomplex, random_state=self.randomstate)
        # n_neighbors=3: 3개와 비교
        self.model4 = KNeighborsTimeSeriesClassifier(n_neighbors=maxneighbor, metric='dtw')
        
        # 라벨 ID 정의
        self.cid = label['circle']
        self.cdl = label['diagonal_left']
        self.cdr = label['diagonal_right']
        self.cho = label['horizontal']
        self.cve = label['vertical']
        
        self.complexlabels = [self.cid, self.cdl, self.cdr]
        self.simplelabels = [self.cho, self.cve]
        self.diagonallabels = [self.cdl, self.cdr]

    def fit(self, x, y, xorig):
        pass # 추론 시 불필요

    def predict(self, x, xorig):
        # x: 테스트 특성 (xtest, 2D)
        # xorig: 테스트 궤적 (xtestorig, 3D 리스트)
        
        ypred1 = self.model1.predict(x)
        ypred = np.zeros(len(x), dtype=int) 

        testsimplemask = (ypred1 == 0)
        testcomplexmask = (ypred1 == 1)

        xtestsimple = x[testsimplemask]
        if xtestsimple.shape[0] > 0:
            ypred2 = self.model2.predict(xtestsimple)
            ypred[testsimplemask] = np.where(ypred2 == 0, self.cho, self.cve)

        xtestcomplex = x[testcomplexmask]
        if xtestcomplex.shape[0] > 0:
            ypred3 = self.model3.predict(xtestcomplex)
            
            # 전체에서 complex로 예측된 인덱스를 찾음
            complex_indices = np.where(testcomplexmask)[0]

            # ypred3 결과를 기반으로 마스크 생성
            mask3circle = (ypred3 == 0)
            mask3diag = (ypred3 == 1)
            
            #complex인덱스 중 circle로 예측된 인덱스
            circle_indices_to_update = complex_indices[mask3circle]
            if len(circle_indices_to_update) > 0:
                ypred[circle_indices_to_update] = self.cid
            
            # complex 중 diag로 예측된 인덱스
            diag_indices_in_subset = complex_indices[mask3diag]
            
            #Model 4 예측 시 3D 궤적 원본 사용
            xtestdiag = [xorig[i] for i in diag_indices_in_subset]

            if len(xtestdiag) > 0:
                xtestdiag = pad_sequences(xtestdiag, padding='post', dtype='float32', value=np.nan)
                ypred4 = self.model4.predict(xtestdiag) # ypred4는 1(DL) 또는 2(DR)
                
                if len(diag_indices_in_subset) == len(ypred4):
                     ypred[diag_indices_in_subset] = ypred4
                else:
                    print("경고: predict 로직에서 길이 불일치 발생")
                
        return ypred

    def getrules(self, featurenames):
        rules1 = export_text(self.model1, feature_names=featurenames, class_names=['simple', 'complex'])
        rules2 = export_text(self.model2, feature_names=featurenames, class_names=['horizontal', 'vertical'])
        rules3 = export_text(self.model3, feature_names=featurenames, class_names=['circle', 'diagonal'])
        # Model 4는 모델 정보를 반환
        rules4 = f"KNeighborsTimeSeriesClassifier(n_neighbors={self.model4.n_neighbors}, metric='{self.model4.metric}')"
        return rules1, rules2, rules3, rules4

modelpath = os.path.join(script_dir, 'decision_tree_add_knn.joblib')
newdata = os.path.join(project_root, 'newdata') 

classnames = list(label.keys())

try:
    model = joblib.load(modelpath)
except IOError:
    print(f"오류: {modelpath}에 모델 없음.")
    exit()

xnew, ytrue = load(newdata) # 증강 안 함

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
    dtw_train_internal_labels = model.model4._y 
    dtw_original_labels_map = model.model4.classes_ 
    # 올바른 테이블 생성
    dtw_train_classnames = [classnames[ dtw_original_labels_map[l] ] for l in dtw_train_internal_labels]
except Exception as e:
    print(f"DTW 설명 라벨 로드 중 오류: {e}")
    dtw_train_classnames = []

for i in range(len(predicted_labels)):
    print(f"샘플 {i+1}: 예측={predicted_labels[i]}, 실제={true_labels[i]}")

    #DTW로 예측된 경우(DL or DR), 왜 그렇게 예측했는지 근거(이웃) 출력
    if predicted_labels[i] in ['diagonal_left', 'diagonal_right']:
        try:
            sample_orig_3d = [xnew[i]]
            sample_padded = pad_sequences(sample_orig_3d, padding='post', dtype='float32', value=np.nan)
            distances, indices = model.model4.kneighbors(sample_padded)
            
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
