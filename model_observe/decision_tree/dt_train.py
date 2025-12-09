#결정나무 모델
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
#데이터 분할
xtrainorig, xtestorig, ytrainorig, ytest = train_test_split(
    xorig, yorig, 
    test_size=testsize, 
    random_state=randomstate, 
    stratify=yorig
)

xauglist, ytrain = augmentdata( 
    xtrainorig, 
    ytrainorig, 
    n=naugment
)

# 훈련 데이터 특성 추출
xtrain, featurenames = extractfeatures(xauglist)
# 테스트 데이터 특성 추출
xtest, _ = extractfeatures(xtestorig)

print(f"훈련 데이터 x {xtrain.shape}, y {ytrain.shape}")
print(f"테스트 데이터 x {xtest.shape}, y {ytest.shape}")

print("Decision Tree 모델 학습")

# max_depth 조절하기
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=randomstate)

model.fit(xtrain, ytrain)

save = 'decision_tree.joblib'
joblib.dump(model, save)
print(f"\n모델 저장 완료: {save}")

print("\n모델 평가")
ypred = model.predict(xtest)
accuracy = np.mean(ypred == ytest)

print(f"\n테스트 정확도: {accuracy:.4f}")

print(classification_report(ytest, ypred, target_names=classnames))

print("모델 설명 (Decision Tree 규칙)")
rules = export_text(model, feature_names=featurenames, class_names=classnames)
print(rules)
