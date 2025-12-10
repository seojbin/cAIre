#계층적 결정나무 모델
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text
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

class HierarchicalClassifier:
    def __init__(self, maxdepthmain=3, maxdepthsimple=3, maxdepthcomplex=3, maxdepthdiag=3, randomstate=42):
        # 모델 하이퍼파라미터
        self.maxdepthmain = maxdepthmain
        self.maxdepthsimple = maxdepthsimple
        self.maxdepthcomplex = maxdepthcomplex
        self.maxdepthdiag = maxdepthdiag
        self.randomstate = randomstate
        
        # 4개모델
        self.model1 = DecisionTreeClassifier(
            max_depth=self.maxdepthmain, random_state=self.randomstate
        )
        self.model2 = DecisionTreeClassifier(
            max_depth=self.maxdepthsimple, random_state=self.randomstate
        )
        self.model3 = DecisionTreeClassifier(
            max_depth=self.maxdepthcomplex, random_state=self.randomstate
        )
        self.model4 = DecisionTreeClassifier(
            max_depth=self.maxdepthdiag, random_state=self.randomstate
        )
        
        self.cid = label['circle']
        self.cdl = label['diagonal_left']
        self.cdr = label['diagonal_right']
        self.cho = label['horizontal']
        self.cve = label['vertical']
        
        self.complexlabels = [self.cid, self.cdl, self.cdr]
        self.simplelabels = [self.cho, self.cve]
        self.diagonallabels = [self.cdl, self.cdr] # L, R 라벨
        self.diag_feature_indices = [0, 3, 6, 9, 15, 18]
        self.diag_feature_names = ['X_mean', 'X_std', 'X_min', 'X_max', 'diff_X', 'range_X']
    def fit(self, x, y):
        
        # Main용 라벨 0=simple, 1=complex
        ytrain1 = np.where(np.isin(y, self.simplelabels), 0, 1)

        # Simple용 데이터/라벨 생성
        simplemask = (ytrain1 == 0)
        xtrain2 = x[simplemask]
        ytrain2_orig = y[simplemask]
        ytrain2 = np.where(ytrain2_orig == self.cho, 0, 1) # 0=h, 1=v

        # Complex
        complexmask = (ytrain1 == 1)
        xtrain3 = x[complexmask]
        ytrain3_orig = y[complexmask]
        ytrain3 = np.where(np.isin(ytrain3_orig, self.diagonallabels), 1, 0)

        # Diagonal L R용
        diagonalmask = np.isin(y, self.diagonallabels)
        # X축 특성만 선택
        xtrain4 = x[diagonalmask][:, self.diag_feature_indices]
        ytrain4_orig = y[diagonalmask]
        ytrain4 = np.where(ytrain4_orig == self.cdl, 0, 1)

        # 4개각각 학습
        self.model1.fit(x, ytrain1)
        self.model2.fit(xtrain2, ytrain2)
        self.model3.fit(xtrain3, ytrain3)
        self.model4.fit(xtrain4, ytrain4)

    def predict(self, x):
        
        # simple/complex 예측
        ypred1 = self.model1.predict(x)
        
        ypred = np.zeros(len(x), dtype=int) 

        # 마스크 생성
        testsimplemask = (ypred1 == 0)
        testcomplexmask = (ypred1 == 1)

        # simple로 예측된 데이터 처리
        xtestsimple = x[testsimplemask]
        if xtestsimple.shape[0] > 0:
            ypred2 = self.model2.predict(xtestsimple)
            ypred[testsimplemask] = np.where(ypred2 == 0, self.cho, self.cve)

        # complex로 예측된 데이터 처리
        xtestcomplex = x[testcomplexmask]
        if xtestcomplex.shape[0] > 0:
            ypred3 = self.model3.predict(xtestcomplex)
            complex_indices = np.where(testcomplexmask)[0]

            # circle/diag 마스크 생성
            mask3circle = (ypred3 == 0)
            mask3diag = (ypred3 == 1)
            #circle
            circle_indices_to_update = complex_indices[mask3circle]
            if len(circle_indices_to_update) > 0:
                ypred[circle_indices_to_update] = self.cid
            
            #diag
            diag_indices_in_subset = complex_indices[mask3diag]
            xtestdiag = xtestcomplex[mask3diag][:, self.diag_feature_indices] # Model 4에 넣을 데이터

            if xtestdiag.shape[0] > 0:
                # diag_l vs diag_r
                ypred4 = self.model4.predict(xtestdiag)
                
                mask4left = (ypred4 == 0)
                mask4right = (ypred4 == 1)

                # left 인덱스
                left_indices_to_update = diag_indices_in_subset[mask4left]
                if len(left_indices_to_update) > 0:
                    ypred[left_indices_to_update] = self.cdl
                
                # right 인덱스
                right_indices_to_update = diag_indices_in_subset[mask4right]
                if len(right_indices_to_update) > 0:
                    ypred[right_indices_to_update] = self.cdr
                
        return ypred

    def getrules(self, featurenames, classnames1, classnames2, classnames3, classnames4):
        rules1 = export_text(self.model1, feature_names=featurenames, class_names=classnames1)
        rules2 = export_text(self.model2, feature_names=featurenames, class_names=classnames2)
        rules3 = export_text(self.model3, feature_names=featurenames, class_names=classnames3)
        rules4 = export_text(self.model4, feature_names=self.diag_feature_names, class_names=['diagonal_left', 'diagonal_right'])
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
#데이터 분할
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

# 훈련 데이터 특성 추출
xtrain, featurenames = extractfeatures(xauglist)
# 테스트 데이터 특성 추출
xtest, _ = extractfeatures(xtestorig)

print(f"훈련 데이터 x {xtrain.shape}, y {ytrainaug.shape}")
print(f"테스트 데이터 x {xtest.shape}, y {ytest.shape}")

model = HierarchicalClassifier(
    maxdepthmain=3,
    maxdepthsimple=3,
    maxdepthcomplex=3,
    maxdepthdiag=3,
    randomstate=randomstate
)

model.fit(xtrain, ytrainaug)

save = 'decision_tree_add.joblib'
joblib.dump(model, save)
print(f"\n모델 저장 완료: {save}")

print("\n모델 평가")
ypred = model.predict(xtest)
accuracy = np.mean(ypred == ytest)

print(f"\n테스트 정확도: {accuracy:.4f}")
print(classification_report(ytest, ypred, target_names=classnames, zero_division=0))

rules1, rules2, rules3, rules4 = model.getrules(
    featurenames,
    classnames1=['simple', 'complex'],
    classnames2=['horizontal', 'vertical'],
    classnames3=['circle', 'diagonal'],
    classnames4=['diagonal_left', 'diagonal_right'])

print("simple vs complex 규칙")
print(rules1)

print(" horizontal vs vertical")
print(rules2)

print("circle vs diagonal")
print(rules3)

print("diagonal_left vs diagonal_right")
print(rules4)
