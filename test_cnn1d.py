import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io

# 인라인 전처리 함수 (훈련과 동일 - import 불필요)
label = {
    'circle': 0,
    'diagonal_left': 1,
    'diagonal_right': 2,
    'horizontal': 3,
    'vertical': 4
}
index = 6
fixed_maxlen = 633  # 훈련 시 패딩 길이 (이전 출력 기준; 필요 시 조정)

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

def pad_fixed(list_, maxlen=fixed_maxlen):
    padx = pad_sequences(list_, maxlen=maxlen, padding='post', dtype='float32', truncating='post')
    print(f"패딩 데이터 형태 {padx.shape}")
    return padx

print("테스트 전처리 로드 완료")

# 모델 로드
try:
    model = load_model(model_path)
    print(f"모델 로드 완료: {model_path}")
    model.summary()  # 구조 확인 (옵션)
except Exception as e:
    print(f"모델 로드 실패: {e}")
    raise

nclasses = len(label)
classnames = list(label.keys())

# ===============================
# 6️⃣ 테스트 데이터 로드 & 전처리
# ===============================
xlist, ytrue = load(data_test)
if len(xlist) == 0:
    raise Exception("테스트 데이터 로드 실패")

xpadded = pad_fixed(xlist)  # 고정 maxlen 패딩

# ===============================
# 7️⃣ 예측 실행
# ===============================
ypredprobs = model.predict(xpadded)
ypred = np.argmax(ypredprobs, axis=1)

# 결과 출력
predicted_labels = [classnames[p] for p in ypred]
true_labels = [classnames[t] for t in ytrue]

print("예측 / 실제 라벨")
print("-"*40)
for i in range(len(predicted_labels)):
    print(f"샘플 {i+1}: 예측={predicted_labels[i]}, 실제={true_labels[i]}")

# ===============================
# 8️⃣ 성능 평가
# ===============================
if len(ytrue) > 0:
    print("\n분류 보고서:")
    print(classification_report(ytrue, ypred, target_names=classnames))
    
    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classnames, yticklabels=classnames)
    plt.title('Test Confusion Matrix (Data#2)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()  # 로컬에서 새 창 팝업
    
    # 정확도 요약
    accuracy = np.mean(ypred == ytrue)
    print(f"\n전체 정확도: {accuracy:.2%} ({np.sum(ypred == ytrue)}/{len(ytrue)} 맞춤)")
else:
    print("테스트 라벨 없음 - 예측만 출력")

# ===============================
# 9️⃣ 단일 샘플 테스트 (옵션: 새 .txt 파일 업로드 후 테스트)
# ===============================
# 예: 새 파일 경로 (Data#2에 test_sample.txt 업로드 가정)
# single_path = os.path.join(project_root, 'Data#2/circle/test_sample.txt')
# single_array = parse(single_path)
# if single_array is not None:
#     single_padded = pad_fixed([single_array])
#     single_pred = model.predict(single_padded)
#     single_class = classnames[np.argmax(single_pred)]
#     print(f"\n단일 샘플 예측: {single_class} (확률: {np.max(single_pred):.2%})")

if __name__ == "__main__":
    pass  # 위 코드가 자동 실행되도록 (main 로직 위에 있음)
