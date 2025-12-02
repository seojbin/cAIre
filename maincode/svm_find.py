import numpy as np
import os
import sys
import contextlib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split  # train_test_split는 필요할 수 있으므로 상위에 둡니다.
from itertools import combinations
from numpy.linalg import norm
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# ==========================================
# 환경 설정 및 상수
# ==========================================
current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label, augmentdata
    from postprocess.feature_extractor import extractfeatures

    BASE_INDICES = {
        'M1': [1, 2, 6, 7, 8, 11],
        'M2': [1, 2, 6, 7, 8, 11],
        'M3': [1, 7, 9, 11, 12],
        'M4': [14, 15]
    }
    ALL_SEARCHABLE_INDICES = [i for i in range(16) if i not in [4, 5]]

    # 전역 상수로 정의하여 NameError 해결
    ITERATIONS = 3
    TRAIN_AUG_N = 9
    ACCURACY_TOLERANCE = 0.01

except ImportError:
    print("오류: postprocess 패키지를 찾을 수 없습니다.")
    exit()

# ★★★★★ 이 변수를 변경하여 M1, M2, M3의 최적화 방향을 설정합니다. ★★★★★
MODE_M1_M3 = 'ADD'


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_full_data():
    """data와 newdata를 모두 로드하여 합친 전체 원본 데이터를 반환합니다."""
    with suppress_stdout():
        data_path = os.path.join(project_root, 'data')
        newdata_path = os.path.join(project_root, 'newdata')
        x_old, y_old = load(data_path)
        x_new, y_new = load(newdata_path)

        if len(y_new) > 0:
            x_raw = x_old + x_new
            y_raw = np.concatenate([y_old, y_new])
        else:
            x_raw = x_old
            y_raw = y_old

    return x_raw, y_raw


def get_hierarchical_data_split(x_raw, y_raw, model_id, is_train=True):
    """모델 ID에 맞게 데이터를 분할하고 이진 레이블을 생성합니다."""
    x_curr, y_curr_orig = [], []
    y_binary = []

    # 데이터 마스킹 로직
    if model_id == 1:
        x_curr = x_raw
        y_curr_orig = y_raw
        y_binary = np.where(y_raw == label['horizontal'], 0, 1)

    elif model_id == 2:
        mask = (y_raw != label['horizontal'])
        x_curr = [x_raw[i] for i in range(len(x_raw)) if mask[i]]
        y_curr_orig = y_raw[mask]
        y_binary = np.where(y_curr_orig == label['vertical'], 0, 1)

    elif model_id == 3:
        targets = [label['circle'], label['diagonal_left'], label['diagonal_right']]
        mask = np.isin(y_raw, targets)
        x_curr = [x_raw[i] for i in range(len(x_raw)) if mask[i]]
        y_curr_orig = y_raw[mask]
        y_binary = np.where(y_curr_orig == label['circle'], 0, 1)

    elif model_id == 4:
        targets = [label['diagonal_left'], label['diagonal_right']]
        mask = np.isin(y_raw, targets)
        x_curr = [x_raw[i] for i in range(len(x_raw)) if mask[i]]
        y_curr_orig = y_raw[mask]
        y_binary = np.where(y_curr_orig == label['diagonal_left'], 0, 1)

    if len(x_curr) == 0:
        return None, None

    if is_train:
        # 학습 시 증강 적용
        with suppress_stdout():
            # NameError를 유발했던 TRAIN_AUG_N 사용
            x_train_aug, y_train_aug_binary = augmentdata(x_curr, y_binary, n=TRAIN_AUG_N)
            x_train_feat_all, _ = extractfeatures(x_train_aug)
        return x_train_feat_all, y_train_aug_binary
    else:
        return None, None


def evaluate_set(X, y, indices):
    if len(indices) < 2 or len(np.unique(y)) < 2:
        return 0.0, 0.0

    X_subset = X[:, indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    clf = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)

    scores = cross_val_score(clf, X_scaled, y, cv=ITERATIONS)
    mean_accuracy = scores.mean()

    clf.fit(X_scaled, y)
    w_norm = norm(clf.coef_)
    margin = 2 / w_norm if w_norm > 1e-6 else 0.0

    return mean_accuracy, margin


# --- 단일 피쳐 변경 최적화 실행 ---
def optimize_single_change(model_id, x_raw, y_raw, feature_names):
    xtrain_feat, ytrain_binary = get_hierarchical_data_split(x_raw, y_raw, model_id, is_train=True)
    if xtrain_feat is None: return {'Action': 'Skipped', 'Indices': []}

    base_indices = BASE_INDICES[f'M{model_id}']
    base_acc, base_margin = evaluate_set(xtrain_feat, ytrain_binary, base_indices)

    best_margin_gain = 0.0
    best_action_desc = "단일 변경으로 마진 증가 없음. 초기 인덱스 유지."
    best_new_indices = base_indices

    # M1, M2, M3 모드 설정에 따른 분기
    if model_id in [1, 2, 3]:

        if MODE_M1_M3 == 'REMOVE':
            candidates = base_indices
            test_loop = 'removal'
        else:  # MODE_M1_M3 == 'ADD'
            candidates = [i for i in ALL_SEARCHABLE_INDICES if i not in base_indices]
            test_loop = 'addition'

        print(f"\n--- M{model_id} ({test_loop.upper()} TEST) | Base Margin: {base_margin:.4f}, Acc: {base_acc:.4f} ---")

        for feature in candidates:
            if test_loop == 'removal':
                test_indices = [i for i in base_indices if i != feature]
                action_desc = f"REMOVE {feature} ({feature_names[feature]})"
            else:  # addition
                test_indices = base_indices + [feature]
                action_desc = f"ADD {feature} ({feature_names[feature]})"

            if len(test_indices) < 2: continue

            acc_test, margin_test = evaluate_set(xtrain_feat, ytrain_binary, test_indices)
            margin_gain = margin_test - base_margin

            # 마진 증가 및 정확도 유지 조건
            if margin_gain > best_margin_gain and acc_test >= (base_acc - ACCURACY_TOLERANCE):
                best_margin_gain = margin_gain
                best_action_desc = action_desc
                best_new_indices = test_indices

    # M4: 단일 피쳐 추가 (고정)
    elif model_id == 4:
        candidates = [i for i in ALL_SEARCHABLE_INDICES if i not in base_indices]

        print(f"\n--- M4 (ADD TEST) | Base Margin: {base_margin:.4f}, Acc: {base_acc:.4f} ---")

        for feature_to_add in candidates:
            test_indices = base_indices + [feature_to_add]
            action_desc = f"ADD {feature_to_add} ({feature_names[feature_to_add]})"

            acc_test, margin_test = evaluate_set(xtrain_feat, ytrain_binary, test_indices)
            margin_gain = margin_test - base_margin

            if margin_gain > best_margin_gain and acc_test >= (base_acc - ACCURACY_TOLERANCE):
                best_margin_gain = margin_gain
                best_action_desc = action_desc
                best_new_indices = test_indices

    return {
        'Model': f"M{model_id}",
        'Initial Indices': base_indices,
        'Initial Margin': base_margin,
        'Action': best_action_desc,
        'New Indices': best_new_indices,
        'Margin Gain': best_margin_gain
    }


def main():
    x_raw, y_raw = get_full_data()

    with suppress_stdout():
        _, feature_names = extractfeatures(x_raw[:2])

    print("=" * 60)
    print(f"🔥 단일 피쳐 변경 마진 최적화 파이프라인 (Mode: {MODE_M1_M3}) 🔥")
    print("=" * 60)

    final_indices = {}

    for m_id in [1, 2, 3, 4]:
        result = optimize_single_change(m_id, x_raw, y_raw, feature_names)

        print(f"\n--- {result['Model']} ---")
        print(f"  Base Margin: {result['Initial Margin']:.4f}")

        if result['Margin Gain'] > 0:
            print(f"  ✅ BEST ACTION: {result['Action']}")
            print(f"  ➡️ Margin Gain: +{result['Margin Gain']:.4f}")
            print(f"  ➡️ New Indices: {result['New Indices']}")
        else:
            print(f"  ⛔ NO GAIN: {result['Action']}")
            print(f"  ➡️ 유지 인덱스: {result['New Indices']}")

        final_indices[f'M{m_id}'] = result['New Indices']

    print("\n" + "=" * 60)
    print("💡 최종 추천 인덱스 (마진 증가 최적):")
    for name, idx in final_indices.items():
        print(f"self.select_indices_{name.lower()} = {idx}")
    print("=" * 60)


if __name__ == "__main__":
    main()