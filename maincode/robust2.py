import numpy as np
import pandas as pd
import sys
import os
import contextlib
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 설정: 비교할 두 가지 피쳐셋 정의
# ==========================================
# Baseline: 기존에 사용하던 설정
# Proposed: 사용자님이 테스트하고 싶은 "추가된" 설정
CONFIGS = {
    "Baseline": {
        "M1": [5, 9, 11, 13, 17, 20,19],
        "M2": [1, 2, 6, 7, 8, 18],
        "M3": [1, 2, 6, 7, 8, 18],
        "M4": [14, 15]
    },
    "Proposed": {
        "M1": [5,9,11, 13, 17, 20,19], 
        "M2": [1, 2, 6, 7, 8, 18],     
        "M3": [1, 2, 6, 7, 8, 18],     
        "M4": [14, 15,13]                
    },
}

ITERATIONS = 30       # 통계적 유의성을 위해 최소 30회 권장
NOISE_MODE = 'LINEAR_DRIFT'
STRESS_LEVEL = 1.0    # 차이가 가장 잘 드러나는 난이도 설정

# ==========================================
# 2. 환경 설정 (기존 코드와 동일)
# ==========================================
current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label, augmentdata, remove_spikes, smooth_trajectory
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("Error: Postprocess package not found. 프로젝트 루트에서 실행해주세요.")
    exit()

label_dict = label # {'circle': 0, ...}

# ==========================================
# 3. 노이즈 함수 (기존 코드 재사용)
# ==========================================
def add_noise_for_test(traj, mode, level):
    new_traj = traj.copy()
    if mode == 'LINEAR_DRIFT':
        drift_dir = np.random.normal(size=(1, 3))
        drift_dir /= np.linalg.norm(drift_dir)
        steps = np.arange(len(traj)).reshape(-1, 1)
        new_traj += steps * (drift_dir * level)
    elif mode == 'DRIFT':
        drift_step = np.random.normal(loc=0.0, scale=level, size=traj.shape)
        new_traj += np.cumsum(drift_step, axis=0)
    # 필요한 다른 모드 추가 가능
    return new_traj

# ==========================================
# 4. 분류기 클래스 (검증용 경량화 버전)
# ==========================================
class HybridVerifier:
    def __init__(self, config_indices):
        self.indices = config_indices
        self.scalers = [StandardScaler() for _ in range(4)]
        self.models = [SVC(kernel='linear', class_weight='balanced', random_state=42) for _ in range(4)]
        
        self.cid = label_dict['circle']
        self.cho = label_dict['horizontal']
        self.cve = label_dict['vertical']
        self.cdl = label_dict['diagonal_left']
        self.cdr = label_dict['diagonal_right']

    def fit(self, x, y):
        # M1
        x1 = x[:, self.indices["M1"]]
        self.scalers[0].fit(x1)
        self.models[0].fit(self.scalers[0].transform(x1), np.where(y == self.cid, 0, 1))
        # M2
        mask2 = y != self.cid
        if np.sum(mask2) > 0:
            x2 = x[mask2][:, self.indices["M2"]]
            self.scalers[1].fit(x2)
            self.models[1].fit(self.scalers[1].transform(x2), np.where(y[mask2] == self.cho, 0, 1))
        # M3
        mask3 = (y != self.cid) & (y != self.cho)
        if np.sum(mask3) > 0:
            x3 = x[mask3][:, self.indices["M3"]]
            self.scalers[2].fit(x3)
            self.models[2].fit(self.scalers[2].transform(x3), np.where(y[mask3] == self.cve, 0, 1))
        # M4
        mask4 = (y != self.cid) & (y != self.cho) & (y != self.cve)
        if np.sum(mask4) > 0:
            x4 = x[mask4][:, self.indices["M4"]]
            self.scalers[3].fit(x4)
            self.models[3].fit(self.scalers[3].transform(x4), np.where(y[mask4] == self.cdl, 0, 1))

    def predict(self, x):
        ypred = np.zeros(len(x), dtype=int)
        
        x1 = x[:, self.indices["M1"]]
        p1 = self.models[0].predict(self.scalers[0].transform(x1))
        ypred[p1 == 0] = self.cid
        
        rest_idx = np.where(p1 == 1)[0]
        if len(rest_idx) == 0: return ypred
        
        x2 = x[rest_idx][:, self.indices["M2"]]
        p2 = self.models[1].predict(self.scalers[1].transform(x2))
        ypred[rest_idx[p2 == 0]] = self.cho
        
        rest_idx2 = rest_idx[p2 == 1]
        if len(rest_idx2) == 0: return ypred
        
        x3 = x[rest_idx2][:, self.indices["M3"]]
        p3 = self.models[2].predict(self.scalers[2].transform(x3))
        ypred[rest_idx2[p3 == 0]] = self.cve
        
        diag_idx = rest_idx2[p3 == 1]
        if len(diag_idx) == 0: return ypred
        
        x4 = x[diag_idx][:, self.indices["M4"]]
        p4 = self.models[3].predict(self.scalers[3].transform(x4))
        for i, val in enumerate(p4):
            ypred[diag_idx[i]] = self.cdl if val == 0 else self.cdr
            
        return ypred

# ==========================================
# 5. 메인 검증 로직
# ==========================================
def main():
    # 데이터 로드
    data_path = os.path.join(project_root, 'data')
    newdata_path = os.path.join(project_root, 'newdata')
    
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        x_old, y_old = load(data_path)
        x_new, y_new = load(newdata_path)
        
    if len(x_new) > 0:
        x_all = x_old + x_new
        y_all = np.concatenate([y_old, y_new])
        test_size = len(x_new)
    else:
        x_all = x_old
        y_all = y_old
        test_size = 20

    print(f"Running Statistical Verification ({ITERATIONS} Iterations)")
    print(f"Mode: {NOISE_MODE}, Level: {STRESS_LEVEL}")
    print("-" * 60)

    sss = StratifiedShuffleSplit(n_splits=ITERATIONS, test_size=test_size, random_state=42)
    
    scores = {'Baseline': [], 'Proposed': []}
    
    for i, (train_idx, val_idx) in enumerate(sss.split(x_all, y_all)):
        # 1. 데이터 분할
        x_train_raw = [x_all[k] for k in train_idx]
        y_train = y_all[train_idx]
        x_val_raw = [x_all[k] for k in val_idx]
        y_val = y_all[val_idx]
        
        # 2. 학습 데이터 증강 및 피쳐 추출
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            x_aug, y_aug = augmentdata(x_train_raw, y_train, n=9)
            x_feat_train, _ = extractfeatures(x_aug)
        
        # 3. 검증 데이터 노이즈 주입 (중요: 두 모델에 *똑같은* 노이즈 데이터를 줘야 공평함)
        x_val_noisy = []
        y_val_expanded = []
        
        for j, traj in enumerate(x_val_raw):
            # 원본
            x_val_noisy.append(traj)
            y_val_expanded.append(y_val[j])
            # 노이즈 복사본 (10개)
            for _ in range(10):
                noisy_traj = add_noise_for_test(traj, NOISE_MODE, STRESS_LEVEL)
                # 전처리 적용
                noisy_traj = remove_spikes(noisy_traj, 5.0)
                noisy_traj = smooth_trajectory(noisy_traj, 3)
                x_val_noisy.append(noisy_traj)
                y_val_expanded.append(y_val[j])
                
        y_val_expanded = np.array(y_val_expanded)
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            x_feat_val, _ = extractfeatures(x_val_noisy)
            
        # 4. 두 모델 평가
        # Baseline
        clf_base = HybridVerifier(CONFIGS["Baseline"])
        clf_base.fit(x_feat_train, y_aug)
        pred_base = clf_base.predict(x_feat_val)
        acc_base = accuracy_score(y_val_expanded, pred_base)
        scores['Baseline'].append(acc_base)
        
        # Proposed
        clf_prop = HybridVerifier(CONFIGS["Proposed"])
        clf_prop.fit(x_feat_train, y_aug)
        pred_prop = clf_prop.predict(x_feat_val)
        acc_prop = accuracy_score(y_val_expanded, pred_prop)
        scores['Proposed'].append(acc_prop)
        
        print(f"Iter {i+1:02d} | Base: {acc_base:.2%} | Prop: {acc_prop:.2%} | Diff: {acc_prop - acc_base:+.2%}")

    # ==========================================
    # 6. 통계적 결론 도출 (T-Test)
    # ==========================================
    base_arr = np.array(scores['Baseline'])
    prop_arr = np.array(scores['Proposed'])
    
    # Paired T-test
    t_stat, p_val = ttest_rel(base_arr, prop_arr)
    mean_diff = np.mean(prop_arr - base_arr)
    
    print("\n" + "="*60)
    print("FINAL STATISTICAL REPORT")
    print("="*60)
    print(f"Mean Accuracy (Baseline): {np.mean(base_arr):.4f} (std: {np.std(base_arr):.4f})")
    print(f"Mean Accuracy (Proposed): {np.mean(prop_arr):.4f} (std: {np.std(prop_arr):.4f})")
    print(f"Average Improvement:      {mean_diff:+.4f} ({mean_diff*100:+.2f}%)")
    print(f"P-value:                  {p_val:.5f}")
    print("-" * 60)
    
    if p_val < 0.05:
        if mean_diff > 0:
            print("Proposed 모델이 더 우수!")
        else:
            print("Proposed 모델이 나쁨")
    else:
        print("차이가 없음")

if __name__ == "__main__":
    main()