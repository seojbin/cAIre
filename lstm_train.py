import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping

try:
    from preprocess import (
        load, 
        augmentdata, 
        pad,
        label
    )
except ImportError:
    print("전처리파일 없음")
    exit()

basepath = './data/'
naugment = 9
testsize = 0.2
randomstate = 42
lstmunits = 64
dropoutrate = 0.3
epochs = 100
batchsize = 32
nclasses = len(label)

print("데이터 전처리")
xorig, yorig = load(basepath)

if len(xorig) == 0:
    print("로드안됨")
    exit()


print("데이터 분할 (원본 기준)")
# 원본 데이터를 훈련용/테스트용 분리
xtrainorig, xtestorig, ytrainorig, ytest = train_test_split(
    xorig, yorig, 
    test_size=testsize, 
    random_state=randomstate, 
    stratify=yorig 
)
print(f"원본 훈련셋: {len(xtrainorig)}개, 원본 테스트셋: {len(xtestorig)}개")

# 훈련용데이터 증강
print("훈련 데이터 증강")
xauglist, ytrain = augmentdata( 
    xtrainorig, 
    ytrainorig, 
    n=naugment 
)

xtestlist = xtestorig
print(f"증강된 훈련셋: {len(xauglist)}개")

#훈련셋과 테스트셋 패딩
ntrain = len(xauglist)

allxlist = xauglist + xtestlist

allxpadded = pad(allxlist)

#패딩된 전체 데이터를 다시 분리
xtrain = allxpadded[:ntrain]
xtest = allxpadded[ntrain:]

print(f"훈련 데이터 x {xtrain.shape}, y {ytrain.shape}")
print(f"테스트 데이터 x {xtest.shape}, y {ytest.shape}")

print("LSTM 모델")

inputshape = (xtrain.shape[1], xtrain.shape[2]) 

model = Sequential([
    Masking(mask_value=0., input_shape=inputshape),
    
    LSTM(lstmunits),
    
    Dropout(dropoutrate),
    
    Dense(nclasses, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("학습 시작")

earlystopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

history = model.fit(
    xtrain, ytrain,
    epochs=epochs,
    batch_size=batchsize,
    validation_data=(xtest, ytest),
    callbacks=[earlystopping],
    verbose=2
)
#모델 저장
MODEL_SAVE_PATH = 'lstm.keras'
model.save(MODEL_SAVE_PATH)
print(f"\n모델 저장 완료: {MODEL_SAVE_PATH}")
print("모델 평가")
loss, accuracy = model.evaluate(xtest, ytest)
print(f"\n테스트 손실 (Loss): {loss:.4f}")
print(f"테스트 정확도 (Accuracy): {accuracy:.4f}")

ypredprobs = model.predict(xtest)
ypred = np.argmax(ypredprobs, axis=1)

classnames = list(label.keys())
print(classification_report(ytest, ypred, target_names=classnames))

cm = confusion_matrix(ytest, ypred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classnames, yticklabels=classnames)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
