import numpy as np
import pandas as pd
import sys
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

# ==========================================
# 환경 설정
# ==========================================
current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label, augmentdata
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("오류: postprocess 패키지를 찾을 수 없습니다.")
    exit()

# 최적화 파라미터 (속도와 정밀도 타협)
ITERATIONS = 10       # 반복 횟수 (기존 50 -> 5로 줄임)
TRAIN_AUG_N = 9      # 학습 증강 배수
VAL_STRESS_N = 6     # 검증용 고강도 노이즈 배수 (기존 3 -> 6)
RANDOM_STATE = 42

def get_data():
    data_path = os.path.join(project_root, 'data')
    newdata_path = os.path.join(project_root, 'newdata')
    
    x_old, y_old = load(data_path)
    x_new, y_new = load(newdata_path)
    
    if len(x_new) > 0:
        x_raw = x_old + x_new
        y_raw = np.concatenate([y_old, y_new])
    else:
        x_raw = x_old
        y_raw = y_old
    return x_raw, y_raw

def evaluate_robustness(model_type, feature_indices, x_raw, y_raw):
    """
    특정 피쳐 조합(feature_indices)에 대한 모델의 강건성 점수 계산
    """
    if len(feature_indices) == 0:
        return 0.0

    # 1. 타겟 및 데이터 필터링
    if model_type == 1: # H vs V vs Complex
        y_target = np.full(y_raw.shape, 2)
        y_target[y_raw == label['horizontal']] = 0
        y_target[y_raw == label['vertical']] = 1
        x_curr = x_raw
        y_curr = y_target
        
    elif model_type == 2: # Circle vs Diagonal (Complex Only)
        targets = [label['circle'], label['diagonal_left'], label['diagonal_right']]
        mask = np.isin(y_raw, targets)
        x_curr = [x_raw[i] for i in range(len(x_raw)) if mask[i]]
        y_temp = y_raw[mask]
        y_curr = np.where(y_temp == label['circle'], 0, 1)
        
    elif model_type == 3: # Left vs Right (Diagonal Only) - SVM 적용
        targets = [label['diagonal_left'], label['diagonal_right']]
        mask = np.isin(y_raw, targets)
        x_curr = [x_raw[i] for i in range(len(x_raw)) if mask[i]]
        y_temp = y_raw[mask]
        y_curr = np.where(y_temp == label['diagonal_left'], 0, 1)

    # 2. 반복 평가 (Stratified Shuffle Split)
    sss = StratifiedShuffleSplit(n_splits=ITERATIONS, test_size=0.3, random_state=RANDOM_STATE)
    scores = []
    
    for train_idx, val_idx in sss.split(x_curr, y_curr):
        # 학습 데이터 준비
        x_train_raw = [x_curr[i] for i in train_idx]
        y_train = y_curr[train_idx]
        
        # 학습 데이터 증강 및 피쳐 추출
        x_train_aug, y_train_aug = augmentdata(x_train_raw, y_train, n=TRAIN_AUG_N)
        x_train_feat_all, _ = extractfeatures(x_train_aug)
        
        # 선택된 피쳐만 사용
        x_train_sel = x_train_feat_all[:, feature_indices]
        
        # 스케일링 및 SVM 학습
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_sel)
        
        clf = SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE)
        clf.fit(x_train_scaled, y_train_aug)
        
        # 검증 (Stress Test)
        fold_correct = 0
        fold_total = 0
        
        for v_idx in val_idx:
            # 원본 검증 데이터 1개 가져오기
            x_val_raw = [x_curr[v_idx]]
            y_val_raw = np.array([y_curr[v_idx]])
            
            # 고강도 노이즈 추가 (Stress Test)
            x_val_aug, y_val_aug = augmentdata(x_val_raw, y_val_raw, n=VAL_STRESS_N)
            x_val_feat_all, _ = extractfeatures(x_val_aug)
            
            x_val_sel = x_val_feat_all[:, feature_indices]
            x_val_scaled = scaler.transform(x_val_sel)
            
            preds = clf.predict(x_val_scaled)
            
            # 노이즈 섞인 버전들이 모두 맞아야 정답 인정 (Strict Robustness)
            if np.all(preds == y_val_aug):
                fold_correct += 1
            fold_total += 1
            
        scores.append(fold_correct / fold_total if fold_total > 0 else 0)
        
    return np.mean(scores)

def greedy_search(model_id, x_raw, y_raw, feature_names):
    print(f"\n>>> [Model {model_id}] Optimizing Features (Greedy Search)...")
    
    selected_indices = []
    remaining_indices = list(range(len(feature_names)))
    best_score = 0.0
    
    while remaining_indices:
        best_idx_round = -1
        best_score_round = -1
        
        # 남은 피쳐 중 하나를 추가했을 때 성능 평가
        for idx in remaining_indices:
            trial_indices = selected_indices + [idx]
            score = evaluate_robustness(model_id, trial_indices, x_raw, y_raw)
            
            if score > best_score_round:
                best_score_round = score
                best_idx_round = idx
        
        # 성능이 향상되거나 유지되면 추가 (Equal도 허용하여 피쳐 확보, 필요시 조건 변경 가능)
        if best_score_round >= best_score:
            selected_indices.append(best_idx_round)
            remaining_indices.remove(best_idx_round)
            best_score = best_score_round
            print(f" + Added '{feature_names[best_idx_round]}' (Score: {best_score:.4f})")
        else:
            # 성능이 떨어지면 중단
            print(f" - Stop: Adding features dropped score to {best_score_round:.4f}")
            break
            
    return selected_indices, best_score

# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    x_raw, y_raw = get_data()
    # 피쳐 이름 추출을 위해 임시 실행
    _, feature_names = extractfeatures(x_raw[:2])
    
    print(f"전체 데이터: {len(y_raw)}개")
    print(f"전체 피쳐({len(feature_names)}개): {feature_names}")
    
    # Model 1, 2, 3 각각 최적화 수행
    res_m1, score_m1 = greedy_search(1, x_raw, y_raw, feature_names)
    res_m2, score_m2 = greedy_search(2, x_raw, y_raw, feature_names)
    res_m3, score_m3 = greedy_search(3, x_raw, y_raw, feature_names)
    
    # 결과 출력 (복사해서 maintrain.py에 사용)
    all_indices = set(range(len(feature_names)))
    
    print("\n\n" + "="*60)
    print("FINAL OPTIMIZED DROP INDICES (Copy to maintrain.py)")
    print("="*60)
    
    print(f"\n[Model 1] H vs V vs Complex (Score: {score_m1:.4f})")
    drop_m1 = sorted(list(all_indices - set(res_m1)))
    print(f"self.drop_indices_model1 = {drop_m1}")
    print(f"Selected: {[feature_names[i] for i in sorted(res_m1)]}")
    
    print(f"\n[Model 2] Circle vs Diagonal (Score: {score_m2:.4f})")
    drop_m2 = sorted(list(all_indices - set(res_m2)))
    print(f"self.drop_indices_model2 = {drop_m2}")
    print(f"Selected: {[feature_names[i] for i in sorted(res_m2)]}")
    
    print(f"\n[Model 3] Left vs Right (SVM Replaced) (Score: {score_m3:.4f})")
    drop_m3 = sorted(list(all_indices - set(res_m3)))
    print(f"self.drop_indices_model3 = {drop_m3}")
    print(f"Selected: {[feature_names[i] for i in sorted(res_m3)]}")
    print("="*60)