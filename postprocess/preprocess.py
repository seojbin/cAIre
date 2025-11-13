import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 클래스와 라벨을 매핑
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
    #단일 궤적 파일 to (N, 3) 형태의 array로 파싱

    try:
        df = pd.read_csv(path, header=None, engine='python', on_bad_lines='skip')
        # r로 시작하는 행만
        if 0 in df.columns:
            df = df[df[0] == 'r'].copy()
        else:
            print(f"{path}에 0번 열이 없음")
            return None
        
        # 7번째 열 NaN 값 행 제거
        df.dropna(subset=[index], inplace=True)
        
        if df.empty:
            print(f"{path}에서 값이 NaN")
            return None
            
        #7번째 열존재 확인
        if index not in df.columns:
            print(f"{path}에 {index}열이 없음")
            return None

        # X/Y/Z 분리하여 리스트로
        series = df[index].astype(str).apply(lambda s: s.split('/'))
        
        # 리스트를 array로

        array = np.array(series.tolist(), dtype=float)

        if array.shape[1] != 3:
            print(f"{path}의 좌표 오류 ({array.shape})")
            return None
            
        return array
        
    except Exception as e:
        print(f"{path} 처리 중 문제 발생 - {e}")
        return None

def load(base):
    #모든 궤적 파일(.txt)파싱, 라벨 할당
    
    # 궤적 배열 저장
    otraject = []
    # 해당 궤적 라벨 저장
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


def augment(traj, strength=1.0):

    #단일 궤적에 Jittering(노이즈 추가)

    newtraj = traj.copy()
    
    # Jittering-(Timesteps, 3) 형태의 정규분포 노이즈 생성
    noise = np.random.normal(loc=0.0, scale=strength, size=newtraj.shape)
    newtraj += noise
    
    # 추가 가능- Scaling, Rotation 등
    
    return newtraj

def augmentdata(origx, origy, n=10):

    #N배 증강된 데이터셋

    print(f"샘플당 {n}개 생성")
    
    augx = []
    augy = []
    
    for traj, label in zip(origx, origy):
        # 1. 원본 데이터를 먼저 추가
        augx.append(traj)
        augy.append(label)
        
        # 2. 증강된 데이터 추가
        for _ in range(n):
            newtraj = augment(traj, strength=1.0) # 1mm 이내
            augx.append(newtraj)
            augy.append(label)
            
    print(f"총 샘플 수: {len(augx)}")
    return augx, np.array(augy)

def pad(list):

    #리스트를 (N, max_len, 3) 패딩
 
    padx = pad_sequences(list, padding='post', dtype='float32', truncating='post')
    print(f"패딩 데이터 형태 {padx.shape}")
    return padx


if __name__ == "__main__":
    
    # 데이터 기본 경로 설정
    base = './'

    origx, origy = load(base)
    
    if len(origx) > 0:

        augx, augy = augmentdata(
            origx, 
            origy, 
            n=19
        )
        
        padx = pad(augx)
        
        print(f"데이터 형태: {padx.shape}")
        print(f"라벨 형태: {augy.shape}")
        print(f"라벨 분포:\n{pd.Series(augy).value_counts().sort_index()}")
        

        np.savez('processed_data.npz', X=padx, y=augy)
        print("\n저장 완료.")

    else:
        print("처리할 데이터를 못찾음")
