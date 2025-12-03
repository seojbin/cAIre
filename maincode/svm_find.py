import numpy as np
import pandas as pd
import sys
import os
import contextlib
import copy
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from itertools import combinations

# ==========================================
# 1. 튜닝 설정 (사용자 입력)
# ==========================================

# [모드 선택] G: Gaussian, S: Spike, D: Drift, R: Rotation
NOISE_MODE_CHAR = 'D'

# [강도 설정]
# G (Gaussian): 표준편차 (예: 50.0)
# S (Spike): 튀는 값의 크기 (예: 500.0)
# D (Drift): 스텝당 밀리는 정도 (예: 3.0 -> 누적되면 큼)
# R (Rotation): 회전 각도 (도, Degree) (예: 15.0 ~ 30.0)
TARGET_NOISE_LEVEL = 7.0

VAL_COPY_N = 20  # 검증 데이터 복제 수

# 모델별 튜닝 방향 (ADD / REMOVE)
TUNING_CONFIG = {
    'M1': 'ADD',
    'M2': 'ADD',
    'M3': 'ADD',
    'M4': 'ADD'
}

# 현재 베이스라인 (최신 최적화 상태)
BASE_INDICES = {
    'M1': [9, 11, 16,18],  # Circle vs Rest
    'M2': [1,2,8,17],  # Horizontal
    'M3': [1,6,7,17],  # Vertical
    'M4': [14,15]  # L vs R
}

# 검색 대상 전체 피쳐 (4:disp, 5:len_disp 제외)
ALL_FEATURES = [i for i in range(19) if i not in [4, 5]]

# ==========================================
# 환경 설정 및 모드 매핑
# ==========================================
NOISE_MAP = {
    'G': 'GAUSSIAN',
    'S': 'SPIKE',
    'D': 'DRIFT',
    'R': 'ROTATION'
}
NOISE_MODE = NOISE_MAP.get(NOISE_MODE_CHAR.upper(), 'GAUSSIAN')

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label, augmentdata
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("[Error] 전처리 패키지(postprocess)를 찾을 수 없습니다.")
    exit()


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old = sys.stdout;
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


# ==========================================
# 현실적 노이즈 생성 함수
# ==========================================
def custom_noise_augment(x_raw, y_raw, n_copies, level, mode):
    aug_x, aug_y = [], []

    for traj, lbl in zip(x_raw, y_raw):
        # 원본 포함
        aug_x.append(traj)
        aug_y.append(lbl)

        for _ in range(n_copies):
            new_traj = traj.copy()

            if mode == 'GAUSSIAN':
                # 모터 전류 노이즈 (전체적으로 떨림)
                noise = np.random.normal(loc=0.0, scale=level, size=traj.shape)
                new_traj += noise

            elif mode == 'SPIKE':
                # 기계 오류 / 통신 튀는 값 (3개 점이 랜덤하게 튐)
                n_points = len(traj)
                n_spikes = 3
                spike_indices = np.random.choice(n_points, n_spikes, replace=False)
                for idx in spike_indices:
                    # 랜덤 방향으로 level만큼 튐
                    direction = np.random.randn(3)
                    new_traj[idx] += direction * level

            elif mode == 'DRIFT':
                # 휴먼 에러 / 자이로 드리프트 (시간이 갈수록 밀림)
                drift_step = np.random.normal(loc=0.0, scale=level, size=traj.shape)
                drift = np.cumsum(drift_step, axis=0)
                new_traj += drift

            elif mode == 'ROTATION':
                # 척추 틀어짐 / 설치 각도 오류 (Z축 기준 회전)
                # level = 각도(degree)
                angle_rad = np.radians(np.random.uniform(-level, level))
                c, s = np.cos(angle_rad), np.sin(angle_rad)
                # Z축 회전 행렬
                R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                new_traj = np.dot(new_traj, R_z.T)

            aug_x.append(new_traj)
            aug_y.append(lbl)

    return aug_x, np.array(aug_y)


# ==========================================
# 데이터 준비
# ==========================================
def prepare_fixed_datasets():
    print(f">>> Data Loading... Noise Mode: {NOISE_MODE} (Level: {TARGET_NOISE_LEVEL})")
    data_path = os.path.join(project_root, 'data')
    newdata_path = os.path.join(project_root, 'newdata')

    with suppress_stdout():
        x_train_raw, y_train_raw = load(data_path)
        x_val_raw, y_val_raw = load(newdata_path)

    if len(x_val_raw) == 0:
        print("[Error] 검증용 newdata가 없습니다.")
        exit()

    # 학습 데이터 (기본 9배 증강)
    with suppress_stdout():
        x_train_aug, y_train_aug = augmentdata(x_train_raw, y_train_raw, n=9)
        x_train_feat, feature_names = extractfeatures(x_train_aug)

    # 검증 데이터 (설정된 현실적 노이즈 적용)
    x_val_stress, y_val_stress = custom_noise_augment(
        x_val_raw, y_val_raw, VAL_COPY_N, TARGET_NOISE_LEVEL, NOISE_MODE
    )

    with suppress_stdout():
        x_val_feat, _ = extractfeatures(x_val_stress)

    print(f"   -> Train Samples: {len(y_train_aug)}")
    print(f"   -> Val Samples: {len(y_val_stress)} (Augmented by {NOISE_MODE})")

    return x_train_feat, y_train_aug, x_val_feat, y_val_stress, feature_names


# ==========================================
# 평가 함수
# ==========================================
def evaluate_model(model_id, indices, x_train, y_train, x_val, y_val):
    if len(indices) < 1: return 0.0

    # 계층적 데이터 필터링
    if model_id == 1:
        mask_t = np.ones(len(y_train), dtype=bool)
        mask_v = np.ones(len(y_val), dtype=bool)
        y_t = np.where(y_train == label['circle'], 0, 1)
        y_v = np.where(y_val == label['circle'], 0, 1)

    elif model_id == 2:
        mask_t = (y_train != label['circle'])
        mask_v = (y_val != label['circle'])
        y_t = np.where(y_train[mask_t] == label['horizontal'], 0, 1)
        y_v = np.where(y_val[mask_v] == label['horizontal'], 0, 1)

    elif model_id == 3:
        mask_t = (y_train != label['circle']) & (y_train != label['horizontal'])
        mask_v = (y_val != label['circle']) & (y_val != label['horizontal'])
        y_t = np.where(y_train[mask_t] == label['vertical'], 0, 1)
        y_v = np.where(y_val[mask_v] == label['vertical'], 0, 1)

    elif model_id == 4:
        diags = [label['diagonal_left'], label['diagonal_right']]
        mask_t = np.isin(y_train, diags)
        mask_v = np.isin(y_val, diags)
        y_t = np.where(y_train[mask_t] == label['diagonal_left'], 0, 1)
        y_v = np.where(y_val[mask_v] == label['diagonal_left'], 0, 1)

    if np.sum(mask_t) == 0 or np.sum(mask_v) == 0: return 0.0

    x_t_sel = x_train[mask_t][:, indices]
    x_v_sel = x_val[mask_v][:, indices]

    scaler = StandardScaler()
    x_t_scaled = scaler.fit_transform(x_t_sel)
    x_v_scaled = scaler.transform(x_v_sel)

    clf = SVC(kernel='linear', class_weight='balanced', random_state=42)
    clf.fit(x_t_scaled, y_t)

    preds = clf.predict(x_v_scaled)
    return accuracy_score(y_v, preds)


# ==========================================
# 메인 루프
# ==========================================
def main():
    xt, yt, xv, yv, fnames = prepare_fixed_datasets()

    print("\n" + "=" * 60)
    print(f" Feature Tuning Start ({NOISE_MODE} Mode)")
    print("=" * 60)

    final_recommendation = {}

    for m_id in [1, 2, 3, 4]:
        name = f"M{m_id}"
        mode = TUNING_CONFIG[name]
        base = BASE_INDICES[name]

        base_acc = evaluate_model(m_id, base, xt, yt, xv, yv)

        print(f"\n[{name}] Mode: {mode} | Base: {base} (Acc: {base_acc * 100:.2f}%)")

        best_acc = base_acc
        best_act = None
        curr_best_indices = base

        if mode == 'REMOVE':
            for feat in base:
                trial = [f for f in base if f != feat]
                if len(trial) < 1: continue
                acc = evaluate_model(m_id, trial, xt, yt, xv, yv)

                if acc > best_acc:
                    print(f"   [OK] Remove {feat} ({fnames[feat]}): {acc * 100:.2f}% (UP)")
                    best_acc = acc
                    best_act = f"Remove {feat}"
                    curr_best_indices = trial
                elif acc == best_acc and len(trial) < len(curr_best_indices):
                    print(f"   [OK] Shape Opt: Remove {feat} ({fnames[feat]}) keeps {acc * 100:.2f}%")
                    curr_best_indices = trial
                    best_act = f"Remove {feat} (Same Acc)"

        elif mode == 'ADD':
            candidates = [f for f in ALL_FEATURES if f not in base]
            for feat in candidates:
                trial = base + [feat]
                acc = evaluate_model(m_id, trial, xt, yt, xv, yv)

                if acc > best_acc:
                    print(f"   [OK] Add {feat} ({fnames[feat]}): {acc * 100:.2f}% (UP)")
                    best_acc = acc
                    best_act = f"Add {feat}"
                    curr_best_indices = trial

        final_recommendation[name] = curr_best_indices
        if best_act:
            print(f"   [BEST] {best_act} -> Final Acc {best_acc * 100:.2f}%")
        else:
            print("   [SKIP] No Improvement.")

    print("\n" + "=" * 60)
    print("Final Recommendations (Copy & Paste):")
    for k, v in final_recommendation.items():
        print(f"self.select_indices_{k.lower()} = {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()