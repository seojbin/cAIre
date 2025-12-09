import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import sys
import os

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)

sys.path.append(project_root)

try:
    from postprocess.preprocess import load, augmentdata, label
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("전처리파일(preprocess.py, feature_extractor.py) 없음")
    exit()

# --- 1. 단순화된 2단계 하이브리드 클래스 (SVM + KNN) ---
class Hybrid_SVM_KNN:
    def __init__(self, maxneighbor=3, randomstate=42):
        self.randomstate = randomstate
        
        # model1: 4-Class (Circle, Diag, H, V) 분류기 (SVM)
        self.model1 = Pipeline([
            ('scaler', StandardScaler()), 
            ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=self.randomstate))
        ])
        
        # model2: Left vs Right 전문가 (KNN-DTW)
        self.model2 = KNeighborsTimeSeriesClassifier(n_neighbors=maxneighbor, metric='dtw', weights='distance')
        
        # 라벨 ID 정의
        self.cid = label['circle']           # 0
        self.cdl = label['diagonal_left']    # 1
        self.cdr = label['diagonal_right']   # 2
        self.cho = label['horizontal']       # 3
        self.cve = label['vertical']         # 4
        
        # 'Diagonal'을 대표하는 새로운 임시 ID
        self.diag_group_id = 5 

    def fit(self, x, y, xorig):
        # x: 훈련 특성 (xtrain, 2D)
        # y: 훈련 라벨 (ytrain, 1D)
        # xorig: 훈련 궤적 (xtrainorig, 3D 리스트)
        
        print("하이브리드 모델 1 (4-Class SVM) 학습...")
        
        # 1. model1 학습용 라벨 생성 (L/R을 'Diagonal 그룹(5)'로 통일)
        y_model1 = np.copy(y)
        mask_diag = np.isin(y, [self.cdl, self.cdr])
        y_model1[mask_diag] = self.diag_group_id 
        
        # model1은 2D 특징(x)과 4-Class 라벨(y_model1)로 학습
        self.model1.fit(x, y_model1)

        print("하이브리드 모델 2 (Diagonal L/R KNN) 학습...")
        # 2. model2 (KNN) 학습 (원본 L/R 라벨(1, 2) 사용)
        xorig_diag = [xorig[i] for i, mask in enumerate(mask_diag) if mask]
        y_diag = y[mask_diag] 
        
        if len(xorig_diag) > 0:
            xorig_diag_padded = pad_sequences(xorig_diag, padding='post', dtype='float32', value=np.nan)
            self.model2.fit(xorig_diag_padded, y_diag)
        
        print("단순화된 하이브리드 모델 학습 완료.")

    def predict(self, x, xorig):
        # x: 테스트 특성 (xtest, 2D)
        # xorig: 테스트 궤적 (xtestorig, 3D 리스트)
        
        # 1. model1 (SVM)로 4-Class 예측 수행
        ypred1 = self.model1.predict(x)
        ypred_final = np.copy(ypred1)

        # 2. model1이 'Diagonal 그룹(5)'으로 예측한 인덱스 찾기
        diag_mask = (ypred1 == self.diag_group_id)
        diag_indices = np.where(diag_mask)[0]

        # 3. 'Diagonal'로 예측된 것이 있다면, model2(KNN)에게 업무 위임
        if len(diag_indices) > 0:
            xtestdiag = [xorig[i] for i in diag_indices]
            
            xtestdiag_padded = pad_sequences(xtestdiag, padding='post', dtype='float32', value=np.nan)
            ypred2_knn = self.model2.predict(xtestdiag_padded) 
            
            ypred_final[diag_indices] = ypred2_knn
            
        return ypred_final

# --- 2. 데이터 준비 및 학습 실행 ---
data = os.path.join(project_root, 'data')
naugment = 9
testsize = 0.2
randomstate = 42
nclasses = len(label)
classnames = list(label.keys())

xorig, yorig = load(data)

if len(xorig) == 0:
    print("로드안됨")
    exit()
    
xtrainorig, xtestorig, ytrainorig, ytest = train_test_split(
    xorig, yorig, 
    test_size=testsize, 
    random_state=randomstate, 
    stratify=yorig
)

xauglist, ytrain = augmentdata(xtrainorig, ytrainorig, n=naugment)
xtrain, featurenames = extractfeatures(xauglist)
xtest, _ = extractfeatures(xtestorig)

print(f"훈련 데이터 x {xtrain.shape}, y {ytrain.shape}")
print(f"테스트 데이터 x {xtest.shape}, y {ytest.shape}")

print("\n단순화된 Hybrid SVM + KNN 모델 학습 시작")
model = Hybrid_SVM_KNN(maxneighbor=3, randomstate=randomstate)

model.fit(xtrain, ytrain, xauglist)

save_filename = 'svm_knn.joblib'
save_path = os.path.join(script_dir, save_filename)
joblib.dump(model, save_path)
print(f"\n모델 저장 완료: {save_path}")

print("\n모델 평가")
ypred = model.predict(xtest, xtestorig)
accuracy = np.mean(ypred == ytest)
print(f"\n테스트 정확도: {accuracy:.4f}")
print(classification_report(ytest, ypred, target_names=classnames))