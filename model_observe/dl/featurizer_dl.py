# postprocess/featurizer_dl.py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample


def parse_xyz_column(xyz_str):
    x, y, z = xyz_str.split('/')
    return np.array([float(x), float(y), float(z)], dtype=np.float32)

def load_trajectory_file(path, col_idx=6):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.strip().split(',')
            xyz = parse_xyz_column(cols[col_idx])
            rows.append(xyz)
    traj = np.stack(rows, axis=0)
    return traj

def resample_traj(traj, target_len=100):
    T = traj.shape[0]
    if T == target_len:
        return traj
    # 각 축별로 리샘플
    resampled = np.stack(
        [resample(traj[:, i], target_len) for i in range(3)],
        axis=1
    )
    return resampled

def normalize_traj(traj):
    # 시작점을 (0,0,0)으로 이동 + 전역 scale 정규화
    shifted = traj - traj[0]                      # translation invariance [file:1]
    scale = np.linalg.norm(shifted, axis=1).max()  # 최대 반경으로 나눔
    if scale < 1e-6:
        scale = 1.0
    normed = shifted / scale
    return normed

#피처 추출

def compute_features(traj):
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    # 기본 통계
    feats = []
    for arr in (x, y, z):
        feats.append(arr.mean())
        feats.append(arr.std())
        feats.append(arr.max() - arr.min())  # range

    # 길이, 변위, ratio
    diffs = np.diff(traj, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    length = seg_len.sum()
    disp_vec = traj[-1] - traj[0]
    disp = np.linalg.norm(disp_vec)
    ratio = length / (disp + 1e-6)

    feats += [length, disp, ratio]

    # 시작-끝 차이
    feats += list(disp_vec.tolist())

    # bounding box 크기
    bbox = traj.max(axis=0) - traj.min(axis=0)
    feats += list(bbox.tolist())

    # xy area
    # x,y의 표준편차 곱으로 근사
    xy_area = x.std() * y.std()
    feats.append(xy_area)

    return np.array(feats, dtype=np.float32)


def load_dataset(root_dir, target_len=100):
    root = Path(root_dir)
    X_list, y_list = [], []
    label_names = sorted([d.name for d in root.iterdir() if d.is_dir()])

    for label_idx, lbl in enumerate(label_names):
        for fpath in (root / lbl).glob('*.txt'):
            # s로 시작하는 정지 데이터는 skip
            if fpath.name.startswith('s'):
                continue
            traj = load_trajectory_file(fpath)
            traj = resample_traj(traj, target_len=target_len)
            traj = normalize_traj(traj)
            feats = compute_features(traj)
            X_list.append(feats)
            y_list.append(label_idx)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y, label_names
