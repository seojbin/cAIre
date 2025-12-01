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
ITERATIONS = 20       # 반복 횟수
TRAIN_AUG_N = 12       # 학습용 증강 배수
INITIAL_STRESS = 6    # 검증용 초기 노이즈
STRESS_STEP = 3       # 1.0 만점 시 늘릴 노이즈 양
MAX_STRESS_ATTEMPTS = 10 # 노이즈 증가 최대 시도 횟수

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
    with suppress_stdout(): 
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
    if len(feature_indices) == 0:
        return 0.0

    if model_type == 1: 
        y_target = np.full(y_raw.shape, 2)
        y_target[y_raw == label['horizontal']] = 0
        y_target[y_raw == label['vertical']] = 1
        x_curr = x_raw
        y_curr = y_target
    elif model_type == 2:
        targets = [label['circle'], label['diagonal_left'], label['diagonal_right']]
        mask = np.isin(y_raw, targets)
        x_curr = [x_raw[i] for i in range(len(x_raw)) if mask[i]]
        y_temp = y_raw[mask]
        y_curr = np.where(y_temp == label['circle'], 0, 1)
    elif model_type == 3: 
        targets = [label['diagonal_left'], label['diagonal_right']]
        mask = np.isin(y_raw, targets)
        x_curr = [x_raw[i] for i in range(len(x_raw)) if mask[i]]
        y_temp = y_raw[mask]
        y_curr = np.where(y_temp == label['diagonal_left'], 0, 1)

    sss = StratifiedShuffleSplit(n_splits=ITERATIONS, test_size=0.3)
    scores = []
    
    for train_idx, val_idx in sss.split(x_curr, y_curr):
        x_train_raw = [x_curr[i] for i in train_idx]
        y_train = y_curr[train_idx]
        
        with suppress_stdout():
            x_train_aug, y_train_aug = augmentdata(x_train_raw, y_train, n=TRAIN_AUG_N)
            x_train_feat_all, _ = extractfeatures(x_train_aug)
        
        x_train_sel = x_train_feat_all[:, feature_indices]
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_sel)
        
        clf = SVC(kernel='linear', C=1.0)
        clf.fit(x_train_scaled, y_train_aug)
        
        fold_correct = 0
        fold_total = 0
        
        for v_idx in val_idx:
            x_val_raw = [x_curr[v_idx]]
            y_val_raw = np.array([y_curr[v_idx]])
            
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
        
        if best_score_round > best_score:
            selected_indices.append(best_idx_round)
            remaining_indices.remove(best_idx_round)
            best_score = best_score_round
        else:
            break
            
    print(f"Done. Best Score: {best_score:.4f}")
    return selected_indices, best_score

def find_best_robust_features(model_id, x_raw, y_raw, feature_names):
    current_stress = INITIAL_STRESS
    history = [] # 모든 시도의 결과를 저장할 리스트
    
    for attempt in range(MAX_STRESS_ATTEMPTS + 1):
        print(f"\n>>> [Model {model_id}] Attempt {attempt+1}/{MAX_STRESS_ATTEMPTS+1} (Noise: {current_stress})")
        
        selected, score = greedy_search(model_id, x_raw, y_raw, feature_names, current_stress)
        
        # 결과 기록 (중간 과정 포함)
        result_record = {
            'stress': current_stress,
            'score': score,
            'selected': selected
        }
        history.append(result_record)
        
        # 1.0 미만이면 한계점이므로 종료
        if score < 1.0:
            print(f"    -> Breaking point found! Score dropped to {score:.4f}.")
            break
        
        # 1.0이면 난이도 올려서 계속 진행
        if attempt < MAX_STRESS_ATTEMPTS:
            print(f"    -> Perfect Score (1.0). Increasing noise level (+{STRESS_STEP})...")
            current_stress += STRESS_STEP
        else:
            print("    -> Max stress reached. Stopping.")
            
    return history # 모든 기록 반환

# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    x_raw, y_raw = get_data()
    with suppress_stdout():
        _, feature_names = extractfeatures(x_raw[:2])
        
    all_indices = set(range(len(feature_names)))
    output_filename = "optimized_params_log.txt"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("Robust Feature Optimization Log (Full History)\n")
        f.write(f"Settings: Epochs={ITERATIONS}, Initial Stress={INITIAL_STRESS}, Step={STRESS_STEP}\n")
        f.write("=================================================\n")

    print(f"전체 데이터: {len(y_raw)}개")
    print(f"전체 피쳐: {feature_names}")
    
    TOTAL_ROUNDS = 5
    
    for round_i in range(1, TOTAL_ROUNDS + 1):
        print(f"\n\n{'='*20} ROUND {round_i}/{TOTAL_ROUNDS} {'='*20}")
        
        # 각 모델별 결과 기록
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"\n[Round {round_i}]\n")
            
        for m_id in [1, 2, 3]:
            # 히스토리 리스트를 받아옴
            history = find_best_robust_features(m_id, x_raw, y_raw, feature_names)
            
            with open(output_filename, "a", encoding="utf-8") as f:
                f.write(f"--- Model {m_id} ---\n")
                
                # 히스토리에 있는 모든 기록을 파일에 저장
                for rec in history:
                    sel = rec['selected']
                    score = rec['score']
                    stress = rec['stress']
                    drop_idxs = sorted(list(all_indices - set(sel)))
                    
                    f.write(f"  [Stress {stress}] Score: {score:.4f}\n")
                    f.write(f"    Selected: {[feature_names[i] for i in sorted(sel)]}\n")
                    f.write(f"    self.drop_indices_model{m_id} = {drop_idxs}\n") # 복사하기 편하게 매번 출력
                    f.write("\n")
                f.write("-" * 30 + "\n")
            
        print(f"\n>>> Round {round_i} Results Saved (All intermediate steps included).")

    print(f"\n\n모든 실험 종료. 결과는 '{output_filename}' 파일을 확인하세요.")