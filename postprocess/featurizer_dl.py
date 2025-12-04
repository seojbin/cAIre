# postprocess/featurizer_dl.py
import numpy as np
from scipy.signal import resample


def resample_traj(traj, target_len=100):
    T = traj.shape[0]
    if T == target_len:
        return traj
    try:
        resampled = np.stack(
            [resample(traj[:, i], target_len) for i in range(3)],
            axis=1
        )
    except ValueError: # 데이터 길이가 너무 짧은 경우 예외 처리
        indices = np.linspace(0, T-1, target_len)
        resampled = np.zeros((target_len, 3))
        for i in range(3):
            resampled[:, i] = np.interp(indices, np.arange(T), traj[:, i])
            
    return resampled

def normalize_traj(traj):
    shifted = traj - traj[0]
    scale = np.linalg.norm(shifted, axis=1).max()
    if scale < 1e-6:
        scale = 1.0
    normed = shifted / scale
    return normed

def extract_features_dl(traj_list, target_len=100):

    processed_data = []
    
    for traj in traj_list:
        if len(traj) < 2: # 너무 짧은 데이터 무시 혹은 패딩
            # 0으로 채운 더미 데이터
            t_processed = np.zeros((target_len, 3))
        else:
            # 1. 길이 통일 (Resample)
            t_res = resample_traj(traj, target_len=target_len)
            # 2. 정규화 (Normalize)
            t_processed = normalize_traj(t_res)
            
        processed_data.append(t_processed)
        
    return np.array(processed_data, dtype=np.float32)
