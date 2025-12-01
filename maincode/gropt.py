import numpy as np
import sys
import os
import contextlib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

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

# 최적화 파라미터
ITERATIONS = 20       # 반복 횟수 (요청대로 20으로 증가)
TRAIN_AUG_N = 9       # 학습용 증강 배수
INITIAL_STRESS = 6    # 검증용 초기 노이즈
STRESS_STEP = 6       # 1.0 만점 시 늘릴 노이즈 양
MAX_STRESS_ATTEMPTS = 5 # 노이즈 증가 최대 시도 횟수 (5번 올려도 1.0이면 포기)

# 로그 숨김용 컨텍스트 매니저
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def get_data():
    with suppress_stdout(): # 데이터 로드 시 잡다한 로그 숨김
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

def evaluate_robustness(model_type, feature_indices, x_raw, y_raw, stress_n):
    """
    특정 피쳐 조합, 특정 스트레스(노이즈) 레벨에서의 강건성 평가
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
        
    elif model_type == 3: # Left vs Right (Diagonal Only) - SVM
        targets = [label['diagonal_left'], label['diagonal_right']]
        mask = np.isin(y_raw, targets)
        x_curr = [x_raw[i] for i in range(len(x_raw)) if mask[i]]
        y_temp = y_raw[mask]
        y_curr = np.where(y_temp == label['diagonal_left'], 0, 1)

    # 2. 반복 평가
    sss = StratifiedShuffleSplit(n_splits=ITERATIONS, test_size=0.3)
    scores = []
    
    for train_idx, val_idx in sss.split(x_curr, y_curr):
        # 학습 데이터 준비
        x_train_raw = [x_curr[i] for i in train_idx]
        y_train = y_curr[train_idx]
        
        # 증강 및 피쳐 추출 (로그 숨김)
        with suppress_stdout():
            x_train_aug, y_train_aug = augmentdata(x_train_raw, y_train, n=TRAIN_AUG_N)
            x_train_feat_all, _ = extractfeatures(x_train_aug)
        
        x_train_sel = x_train_feat_all[:, feature_indices]
        
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_sel)
        
        clf = SVC(kernel='linear', C=1.0)
        clf.fit(x_train_scaled, y_train_aug)
        
        # 검증 (Stress Test)
        fold_correct = 0
        fold_total = 0
        
        for v_idx in val_idx:
            x_val_raw = [x_curr[v_idx]]
            y_val_raw = np.array([y_curr[v_idx]])
            
            # 지정된 Stress 레벨로 노이즈 추가 (로그 숨김)
            with suppress_stdout():
                x_val_aug, y_val_aug = augmentdata(x_val_raw, y_val_raw, n=stress_n)
                x_val_feat_all, _ = extractfeatures(x_val_aug)
            
            x_val_sel = x_val_feat_all[:, feature_indices]
            x_val_scaled = scaler.transform(x_val_sel)
            
            preds = clf.predict(x_val_scaled)
            
            if np.all(preds == y_val_aug):
                fold_correct += 1
            fold_total += 1
            
        scores.append(fold_correct / fold_total if fold_total > 0 else 0)
        
    return np.mean(scores)

def greedy_search(model_id, x_raw, y_raw, feature_names, stress_n):
    """
    Greedy Search 수행. 성능 향상이 없으면 즉시 중단.
    """
    print(f"   [Search] Stress Level={stress_n} ... ", end='', flush=True)
    selected_indices = []
    remaining_indices = list(range(len(feature_names)))
    best_score = 0.0
    
    while remaining_indices:
        best_idx_round = -1
        best_score_round = -1
        
        for idx in remaining_indices:
            trial_indices = selected_indices + [idx]
            score = evaluate_robustness(model_id, trial_indices, x_raw, y_raw, stress_n)
            
            if score > best_score_round:
                best_score_round = score
                best_idx_round = idx
        
        # [Strict Stop] 성능이 '확실히' 오르지 않으면(초과) 중단
        if best_score_round > best_score:
            selected_indices.append(best_idx_round)
            remaining_indices.remove(best_idx_round)
            best_score = best_score_round
        else:
            # 성능이 같거나 떨어지면 더 이상 추가하지 않음
            break
            
    print(f"Done. Best Score: {best_score:.4f}")
    return selected_indices, best_score

def find_best_robust_features(model_id, x_raw, y_raw, feature_names):
    """
    Adaptive Stress Logic:
    1.0점이 나오면 Stress를 높여서 다시 검사.
    점수가 1.0 미만으로 떨어지거나, 최대 시도 횟수를 넘기면 종료.
    """
    current_stress = INITIAL_STRESS
    final_selected = []
    final_score = 0.0
    
    for attempt in range(MAX_STRESS_ATTEMPTS + 1):
        print(f"\n>>> [Model {model_id}] Attempt {attempt+1}/{MAX_STRESS_ATTEMPTS+1} (Noise: {current_stress})")
        
        selected, score = greedy_search(model_id, x_raw, y_raw, feature_names, current_stress)
        
        final_selected = selected
        final_score = score
        
        # 만점이 아니면(변별력이 생겼으면) 현재 결과 확정
        if score < 1.0:
            print(f"    -> Breaking point found! Score dropped to {score:.4f}.")
            break
        
        # 만점이면 더 어려운 난이도로 도전
        if attempt < MAX_STRESS_ATTEMPTS:
            print(f"    -> Perfect Score (1.0). Increasing noise level (+{STRESS_STEP})...")
            current_stress += STRESS_STEP
        else:
            print("    -> Max stress reached. Stopping with score 1.0.")
            
    return final_selected, final_score, current_stress

# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    x_raw, y_raw = get_data()
    # 피쳐 이름 추출용 (로그 숨김)
    with suppress_stdout():
        _, feature_names = extractfeatures(x_raw[:2])
        
    all_indices = set(range(len(feature_names)))
    
    output_filename = "optimized_params_log.txt"
    
    # 파일 초기화
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("Robust Feature Optimization Log (5 Iterations)\n")
        f.write(f"Settings: Epochs={ITERATIONS}, Initial Stress={INITIAL_STRESS}, Step={STRESS_STEP}\n")
        f.write("=================================================\n")

    print(f"전체 데이터: {len(y_raw)}개")
    print(f"전체 피쳐: {feature_names}")
    
    TOTAL_ROUNDS = 5
    
    for round_i in range(1, TOTAL_ROUNDS + 1):
        print(f"\n\n{'='*20} ROUND {round_i}/{TOTAL_ROUNDS} {'='*20}")
        
        results = {}
        for m_id in [1, 2, 3]:
            sel, score, stress = find_best_robust_features(m_id, x_raw, y_raw, feature_names)
            drop_idxs = sorted(list(all_indices - set(sel)))
            results[m_id] = {
                'score': score,
                'stress': stress,
                'selected': [feature_names[i] for i in sorted(sel)],
                'drop': drop_idxs
            }
        
        # 파일 기록
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"\n[Round {round_i}]\n")
            for m_id in [1, 2, 3]:
                res = results[m_id]
                f.write(f"Model {m_id} (Stress: {res['stress']}, Score: {res['score']:.4f})\n")
                f.write(f"  Selected: {res['selected']}\n")
                f.write(f"  Drop Indices: {res['drop']}\n")
                f.write(f"  self.drop_indices_model{m_id} = {res['drop']}\n\n")
            
        print(f"\n>>> Round {round_i} Results Saved.")

    print(f"\n\n모든 실험 종료. 결과는 '{output_filename}' 파일을 확인하세요.")