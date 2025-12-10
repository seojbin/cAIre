import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys
import os
from sklearn.tree import DecisionTreeClassifier 

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

class HierarchicalClassifier:
    def __init__(self, maxdepthmain=3, maxdepthsimple=3, maxdepthcomplex=3, maxdepthdiag=3, randomstate=42):
        # 모델 하이퍼파라미터
        self.maxdepthmain = maxdepthmain
        self.maxdepthsimple = maxdepthsimple
        self.maxdepthcomplex = maxdepthcomplex
        self.maxdepthdiag = maxdepthdiag
        self.randomstate = randomstate
        
        # 4개의 내부 모델
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
        pass

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

        # complex 처리
        xtestcomplex = x[testcomplexmask]
        if xtestcomplex.shape[0] > 0:
            # circle vs diag
            ypred3 = self.model3.predict(xtestcomplex)
            complex_indices = np.where(testcomplexmask)[0]

            # circle/diag 마스크 생성
            mask3circle = (ypred3 == 0)
            mask3diag = (ypred3 == 1)
            
            # circle 예측
            circle_indices_to_update = complex_indices[mask3circle]
            if len(circle_indices_to_update) > 0:
                ypred[circle_indices_to_update] = self.cid
            
            #complex 중 diag 예측
            diag_indices_in_subset = complex_indices[mask3diag]
            xtestdiag = xtestcomplex[mask3diag][:, self.diag_feature_indices] # Model 4에 넣을 데이터

            if xtestdiag.shape[0] > 0:
                # diag_l vs diag_r
                ypred4 = self.model4.predict(xtestdiag)
                
                mask4left = (ypred4 == 0)
                mask4right = (ypred4 == 1)

                # left 예측
                left_indices_to_update = diag_indices_in_subset[mask4left]
                if len(left_indices_to_update) > 0:
                    ypred[left_indices_to_update] = self.cdl
                
                # right 예측
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

modelpath = os.path.join(script_dir, 'decision_tree_add.joblib')
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

ypred = model.predict(xnewfeatures)

print("추론 결과")
predicted_labels = [classnames[p] for p in ypred]
true_labels = [classnames[t] for t in ytrue]

for i in range(len(predicted_labels)):
    print(f"샘플 {i+1}: 예측={predicted_labels[i]}, 실제={true_labels[i]}")

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
