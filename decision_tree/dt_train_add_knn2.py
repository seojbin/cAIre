#하이브리드 결정나무 모델
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesResampler
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

class HybridClassifier:
    def __init__(self,maxneighbor1=5,maxdepthsimple=3,maxdepthcomplex=3,maxneighbor2=3,randomstate=42):
        self.randomstate = randomstate
        self.scaler = StandardScaler()
        self.model1 = KNeighborsClassifier(n_neighbors=maxneighbor1) 
        
        self.model2 = DecisionTreeClassifier(max_depth=maxdepthsimple, random_state=self.randomstate)
        self.model3 = DecisionTreeClassifier(max_depth=maxdepthcomplex, random_state=self.randomstate)
        self.model4 = KNeighborsTimeSeriesClassifier(n_neighbors=maxneighbor2, metric='dtw')
        
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
        # x: 전체 훈련 특성 (xtrain, 2D)
        # y: 전체 훈련 라벨 (ytrainaug, 1D)
        # xorig: 전체 훈련 궤적 (xauglist, 3D 리스트)
        
        self.scaler.fit(x)
        x_scaled = self.scaler.transform(x)
        
        # Main용 라벨 생성
        ytrain1 = np.where(np.isin(y, self.simplelabels), 0, 1)

        # Simple용 데이터/라벨 생성
        simplemask = (ytrain1 == 0)
        xtrain2 = x[simplemask]
        ytrain2_orig = y[simplemask]
        ytrain2 = np.where(ytrain2_orig == self.cho, 0, 1) 

        # Complex용 데이터/라벨 생성
        complexmask = (ytrain1 == 1)
        xtrain3 = x[complexmask]
        ytrain3_orig = y[complexmask]
        ytrain3 = np.where(np.isin(ytrain3_orig, self.diagonallabels), 1, 0)

        #Diagonal용 데이터/라벨 생성
        diagonalmask = np.isin(y, self.diagonallabels)
        xtrain4 = [xorig[i] for i in range(len(xorig)) if diagonalmask[i]]
        xtrain4 = pad_sequences(xtrain4, padding='post', dtype='float32', value=np.nan)
        ytrain4 = y[diagonalmask] 

        # 4개 모델 각각 학습
        self.model1.fit(x_scaled, ytrain1)
        self.model2.fit(xtrain2, ytrain2)
        self.model3.fit(xtrain3, ytrain3)
        self.model4.fit(xtrain4, ytrain4)

    def predict(self, x, xorig):
        # x: 테스트 특성 (xtest, 2D)
        # xorig: 테스트 궤적 (xtestorig, 3D 리스트)
        
        x_scaled = self.scaler.transform(x)
        
        ypred1 = self.model1.predict(x_scaled) # 스케일링된 데이터로 예측
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
            
            complex_indices = np.where(testcomplexmask)[0]
            mask3circle = (ypred3 == 0)
            mask3diag = (ypred3 == 1)
            
            circle_indices_to_update = complex_indices[mask3circle]
            if len(circle_indices_to_update) > 0:
                ypred[circle_indices_to_update] = self.cid
            
            diag_indices_in_subset = complex_indices[mask3diag]
            xtestdiag = [xorig[i] for i in diag_indices_in_subset]

            if len(xtestdiag) > 0:
                xtestdiag = pad_sequences(xtestdiag, padding='post', dtype='float32', value=np.nan)
                ypred4 = self.model4.predict(xtestdiag) 
                
                if len(diag_indices_in_subset) == len(ypred4):
                     ypred[diag_indices_in_subset] = ypred4
                else:
                    print("경고: predict 로직에서 길이 불일치 발생")
                
        return ypred

    def getrules(self, featurenames):
        rules1 = f"KNeighborsClassifier(n_neighbors={self.model1.n_neighbors}, metric='euclidean')"
        rules2 = export_text(self.model2, feature_names=featurenames, class_names=['horizontal', 'vertical'])
        rules3 = export_text(self.model3, feature_names=featurenames, class_names=['circle', 'diagonal'])
        rules4 = f"KNeighborsTimeSeriesClassifier(n_neighbors={self.model4.n_neighbors}, metric='{self.model4.metric}')"
        return rules1, rules2, rules3, rules4

data = os.path.join(project_root, 'data')
naugment = 9
testsize = 0.2
randomstate = 42
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

xauglist, ytrainaug = augmentdata( 
    xtrainorig, 
    ytrainorig, 
    n=naugment
)

xtrain, featurenames = extractfeatures(xauglist)
xtest, _ = extractfeatures(xtestorig)

print(f"훈련 데이터 x {xtrain.shape}, y {ytrainaug.shape}")
print(f"테스트 데이터 x {xtest.shape}, y {ytest.shape}")

model = HybridClassifier(maxneighbor1=5,maxdepthsimple=3,maxdepthcomplex=3,maxneighbor2=3,randomstate=42)

model.fit(xtrain, ytrainaug, xauglist)

save = 'decision_tree_add_knn.joblib'
joblib.dump(model, save)
print(f"\n모델 저장 완료: {save}")

print("\n모델 평가")
ypred = model.predict(xtest,xtestorig)
accuracy = np.mean(ypred == ytest)

print(f"\n테스트 정확도: {accuracy:.4f}")
print(classification_report(ytest, ypred, target_names=classnames, zero_division=0))

rules1, rules2, rules3, rules4 = model.getrules(featurenames)

print("simple vs complex")
print(rules1)

print(" horizontal vs vertical")
print(rules2)

print("circle vs diagonal")
print(rules3)

print("diagonal_left vs diagonal_right")
print(rules4)