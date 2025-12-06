import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io

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


def remove_spikes(traj, threshold_std=3.0):
    """
    연속된 두 점 사이의 거리가 평균 거리보다 비정상적으로 클 경우(튀는 값) 제거
    """
    if len(traj) < 3: return traj

    # 각 점 사이의 유클리드 거리 계산
    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    mean_dist = np.mean(diffs)
    std_dist = np.std(diffs)

    # 임계값 설정 (평균 + N * 표준편차)
    limit = mean_dist + (threshold_std * std_dist) + 1e-6  # 0.0 방지

    # 튀는 구간(Jump)이 있는지 마스킹
    # 첫 점은 유지(True), 이후 점들은 거리가 limit보다 작아야 유지
    mask = [True]
    for d in diffs:
        if d < limit:
            mask.append(True)
        else:
            mask.append(False)  # 튀는 점 제거

    return traj[mask]


def smooth_trajectory(traj, window_size=3):
    """
    이동 평균 필터(Moving Average)를 사용하여 궤적을 부드럽게 만듦 (노이즈 제거)
    """
    if len(traj) < window_size: return traj

    df = pd.DataFrame(traj)
    # 중심을 기준으로 이동 평균, 양끝은 원본 유지 혹은 fill
    smoothed = df.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed.values


def parse(path):
    # 단일 궤적 파일 to (N, 3) 형태의 array로 파싱
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        r_lines = [line for line in lines if line.startswith('r,')]

        if not r_lines:
            return None

        cleaned_data = "".join(r_lines)
        df = pd.read_csv(io.StringIO(cleaned_data), header=None, engine='python')

        if index not in df.columns:
            return None

        df.dropna(subset=[index], inplace=True)
        if df.empty: return None

        series = df[index].astype(str).apply(lambda s: s.split('/'))

        # 데이터가 비어있거나 형식이 안 맞는 경우 예외처리
        valid_rows = series.apply(lambda x: len(x) == 3)
        series = series[valid_rows]

        if len(series) == 0: return None

        array = np.array(series.tolist(), dtype=float)
        return array

    except Exception as e:
        print(f"Error parsing {path}: {e}")
        return None


def load(base):
    otraject = []
    olabels = []

    for cname, cid in label.items():
        path = os.path.join(base, cname)
        if not os.path.isdir(path):
            print(f"Skipping {path} (Not found)")
            continue

        print(f"Loading {cname}...")
        files = glob.glob(os.path.join(path, "*.txt"))

        count = 0
        for file in files:
            array = parse(file)
            if array is not None and len(array) > 0:
                otraject.append(array)
                olabels.append(cid)
                count += 1
        print(f" - {count} samples loaded.")

    return otraject, np.array(olabels)


def augment(traj, strength=1.0, scale_r=(0.9, 1.1), offset_mm=1.0):
    newtraj = traj.copy()
    n_points = len(newtraj)
    # 1. Gaussian Noise
    noise = np.random.normal(loc=0.0, scale=strength, size=newtraj.shape)
    newtraj += noise

    # 2. Scaling
    if np.random.rand() > 0.3:
        scale = np.random.uniform(scale_r[0], scale_r[1])
        newtraj *= scale

    # 3. Offset (Shift)
    if np.random.rand() > 0.3:
        offset = np.random.uniform(-offset_mm, offset_mm, size=3)
        newtraj += offset

    # 4. Rotation 
    if np.random.rand() > 0.3:
        # -15도 ~ +15도 회전
        theta = np.radians(np.random.uniform(-15, 15))
        c, s = np.cos(theta), np.sin(theta)
        # Z축 회전 행렬
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        newtraj = np.dot(newtraj, R.T)

    if np.random.rand() > 0.5:
        drift_level = strength * 0.5
        drift_step = np.random.normal(loc=0.0, scale=drift_level, size=newtraj.shape)
        drift = np.cumsum(drift_step, axis=0)
        newtraj += drift
    
    if np.random.rand() > 0.7:
        distort_mag = strength * 20.0 
        drift_dir = np.random.normal(size=(1, 3))
        drift_dir /= (np.linalg.norm(drift_dir) + 1e-6)
        steps = np.linspace(0, 1, n_points).reshape(-1, 1)
        steps = steps ** 2
        linear_distortion = steps * (drift_dir * distort_mag)
        newtraj += linear_distortion
    if np.random.rand() > 0.5:
        idxs = sorted(np.random.choice(len(newtraj), int(len(newtraj)*0.9), replace=False))
        newtraj = newtraj[idxs]

    return newtraj


def augmentdata(origx, origy, n=10):
    print(f"Augmenting data x{n}...")
    augx = []
    augy = []

    for traj, label in zip(origx, origy):
        augx.append(traj)
        augy.append(label)

        for _ in range(n):
            # 노이즈 강도를 약간씩 다르게 줄 수도 있음
            newtraj = augment(traj, strength=1.0, scale_r=(0.85, 1.15), offset_mm=2.0)
            augx.append(newtraj)
            augy.append(label)

    return augx, np.array(augy)


def pad(list_data):
    return pad_sequences(list_data, padding='post', dtype='float32', truncating='post')


if __name__ == "__main__":
    base = './'  
    origx, origy = load(base)

    if len(origx) > 0:
        augx, augy = augmentdata(origx, origy, n=9)

        print(f"Total Samples: {len(augx)}")

        X_save = np.array(augx, dtype=object)

        np.savez('processed_data.npz', X=X_save, y=augy)
        print("Data saved to processed_data.npz")
    else:
        print("No data found.")