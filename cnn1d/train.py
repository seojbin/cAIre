import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import glob
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io

# 클래스와 라벨 매핑
label = {
    'circle': 0,
    'diagonal_left': 1,
    'diagonal_right': 2,
    'horizontal': 3,
    'vertical': 4
}

# 열 인덱스
index = 6

def parse(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        r_lines = [line for line in lines if line.startswith('r,')]
        if not r_lines:
            print(f"{path}에 'r'로 시작하는 데이터 행이 없음")
            return None
        cleaned_data = "".join(r_lines)
        df = pd.read_csv(io.StringIO(cleaned_data), header=None, engine='python')
        if index not in df.columns:
            print(f"{path}에 {index}열이 없음 (파일 형식 오류)")
            return None
        df.dropna(subset=[index], inplace=True)
        if df.empty:
            print(f"{path}에서 값이 NaN")
            return None
        series = df[index].astype(str).apply(lambda s: s.split('/'))
        array = np.array(series.tolist(), dtype=float)
        if array.shape[1] != 3:
            print(f"{path}의 좌표 오류 ({array.shape})")
            return None
        return array
    except Exception as e:
        print(f"{path} 처리 중 문제 발생 - {e}")
        return None

def load(base):
    otraject = []
    olabels = []
    for cname, cid in label.items():
        path = os.path.join(base, cname)
        if not os.path.isdir(path):
            print(f"'{path}' 폴더 없음")
            continue
        print(f"{cname}(라벨 {cid})")
        files = glob.glob(os.path.join(path, "*.txt"))
        for file in files:
            array = parse(file)
            if array is not None and len(array) > 0:
                otraject.append(array)
                olabels.append(cid)
    print(f"{len(otraject)}개의 샘플")
    return otraject, np.array(olabels)

def augment(traj, strength=1.0, scale_r=(0.9, 1.1), offset_mm=1.0):
    newtraj = traj.copy()
    noise = np.random.normal(loc=0.0, scale=strength, size=newtraj.shape)
    newtraj += noise
    # 크기 조절·평행이동 생략 (주석 처리)
    return newtraj

def augmentdata(origx, origy, n=10):
    print(f"샘플당 {n}개 생성")
    augx = []
    augy = []
    for traj, label_val in zip(origx, origy):
        augx.append(traj)
        augy.append(label_val)
        for _ in range(n):
            newtraj = augment(traj, strength=1.0, scale_r=(0.9, 1.1), offset_mm=1.0)
            augx.append(newtraj)
            augy.append(label_val)
    print(f"총 샘플 수: {len(augx)}")
    return augx, np.array(augy)

def pad_fixed(list_, maxlen=None):
    if maxlen is None:
        padx = pad_sequences(list_, padding='post', dtype='float32', truncating='post')
    else:
        padx = pad_sequences(list_, maxlen=maxlen, padding='post', dtype='float32', truncating='post')
    print(f"패딩 데이터 형태 {padx.shape}")
    return padx

if __name__ == "__main__":
    
    data_train = './'  # 훈련 데이터 (현재 디렉토리 기준)
    data_test = './'   # 테스트 데이터 (현재 디렉토리 기준)
    model_save_path = './cnn1d.keras'  # 모델 저장 (현재 디렉토리)
    
    naugment = 9
    testsize = 0.2
    randomstate = 42
    dropoutrate = 0.3
    epochs = 100
    batchsize = 32
    nclasses = len(label)
    classnames = list(label.keys())

    np.random.seed(randomstate)
    tf.random.set_seed(randomstate)

    print("인라인 전처리 로드 완료")

    # 훈련 데이터 로드
    xorig, yorig = load(data_train)
    if len(xorig) == 0:
        raise Exception("훈련 데이터 로드 실패")

    # 원본 데이터 분할 및 증강
    xtrainorig, xtestorig, ytrainorig, ytest = train_test_split(
        xorig, yorig, test_size=testsize, random_state=randomstate, stratify=yorig
    )
    xauglist, ytrain = augmentdata(xtrainorig, ytrainorig, n=naugment)
    xtestlist = xtestorig
    ntrain = len(xauglist)
    allxlist = xauglist + xtestlist
    allxpadded = pad_fixed(allxlist)
    fixed_maxlen = allxpadded.shape[1]
    xtrain = allxpadded[:ntrain]
    xtest = allxpadded[ntrain:]

    print(f"훈련 데이터 x {xtrain.shape}, y {ytrain.shape}")
    print(f"테스트 데이터 x {xtest.shape}, y {ytest.shape}")

    # 모델 정의
    inputshape = (fixed_maxlen, 3)
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=inputshape),
        MaxPooling1D(2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(dropoutrate),
        Dense(nclasses, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 학습
    earlystopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        xtrain, ytrain,
        epochs=epochs,
        batch_size=batchsize,
        validation_data=(xtest, ytest),
        callbacks=[earlystopping],
        verbose=2
    )

    # 학습 시각화
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    model.save(model_save_path)
    print(f"\n모델 저장 완료: {model_save_path}")

    # 테스트 데이터 로드 및 예측
    xlist, ytrue = load(data_test)
    if len(xlist) == 0:
        raise Exception("테스트 데이터 로드 실패")
    xpadded = pad_fixed(xlist, maxlen=fixed_maxlen)
    ypredprobs = model.predict(xpadded)
    ypred = np.argmax(ypredprobs, axis=1)
    predicted_labels = [classnames[p] for p in ypred]
    true_labels = [classnames[t] for t in ytrue]
    print("예측 / 실제 라벨")
    print("-"*40)
    for i in range(len(predicted_labels)):
        print(f"샘플 {i+1}: 예측={predicted_labels[i]}, 실제={true_labels[i]}")
    if len(ytrue) > 0:
        print(classification_report(ytrue, ypred, target_names=classnames))
        cm = confusion_matrix(ytrue, ypred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classnames, yticklabels=classnames)
        plt.title('Data Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
