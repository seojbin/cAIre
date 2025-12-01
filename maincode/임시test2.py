import numpy as np
import sys
import os
import contextlib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label, augmentdata
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    exit()

ITERATIONS = 20
TRAIN_AUG_N = 12
INITIAL_STRESS = 6
STRESS_STEP = 3
MAX_STRESS_ATTEMPTS = 10

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
    if len(feature_indices) == 0: return 0.0

    x_curr, y_curr = [], []

    # 1. Horizontal vs Rest
    if model_type == 1:
        x_curr = x_raw
        y_curr = np.where(y_raw == label['horizontal'], 0, 1)

    # 2. Vertical vs Complex (Circle, Diagonals)
    elif model_type == 2:
        mask = (y_raw != label['horizontal'])
        x_curr = [x_raw[i] for i in range(len(x_raw)) if mask[i]]
        y_temp = y_raw[mask]
        y_curr = np.where(y_temp == label['vertical'], 0, 1)

    # 3. Circle vs Diagonals
    elif model_type == 3:
        targets = [label['circle'], label['diagonal_left'], label['diagonal_right']]
        mask = np.isin(y_raw, targets)
        x_curr = [x_raw[i] for i in range(len(x_raw)) if mask[i]]
        y_temp = y_raw[mask]
        y_curr = np.where(y_temp == label['circle'], 0, 1)

    if len(x_curr) == 0: return 0.0

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
        
        clf = SVC(kernel='linear', C=1.0, class_weight='balanced')
        clf.fit(x_train_scaled, y_train_aug)
        
        fold_y_true = []
        fold_y_pred = []
        
        for v_idx in val_idx:
            x_val_raw = [x_curr[v_idx]]
            y_val_raw = np.array([y_curr[v_idx]])
            
            with suppress_stdout():
                x_val_aug, y_val_aug = augmentdata(x_val_raw, y_val_raw, n=stress_n)
                x_val_feat_all, _ = extractfeatures(x_val_aug)
            
            x_val_sel = x_val_feat_all[:, feature_indices]
            x_val_scaled = scaler.transform(x_val_sel)
            preds = clf.predict(x_val_scaled)
            
            fold_y_true.extend(y_val_aug)
            fold_y_pred.extend(preds)
            
        score = f1_score(fold_y_true, fold_y_pred, average='macro', zero_division=0)
        scores.append(score)
        
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
    history = []
    
    for attempt in range(MAX_STRESS_ATTEMPTS + 1):
        print(f"\n>>> [Model {model_id}] Attempt {attempt+1} (Noise: {current_stress})")
        selected, score = greedy_search(model_id, x_raw, y_raw, feature_names, current_stress)
        
        history.append({'stress': current_stress, 'score': score, 'selected': selected})
        
        if score < 1.0:
            print(f"    -> Breaking point found! Score: {score:.4f}")
            break
        
        if attempt < MAX_STRESS_ATTEMPTS:
            current_stress += STRESS_STEP
        else:
            pass
            
    return history

if __name__ == "__main__":
    x_raw, y_raw = get_data()
    with suppress_stdout():
        _, feature_names = extractfeatures(x_raw[:2])
    all_indices = set(range(len(feature_names)))
    output_filename = "optimized_params_log.txt"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("Hierarchy 4-Step Robust Optimization\n")
        f.write("Model 1: Horizontal vs Rest\n")
        f.write("Model 2: Vertical vs Complex\n")
        f.write("Model 3: Circle vs Diagonal\n")
        f.write("Model 4: (KNN-DTW used, no feature opt)\n")
        f.write("=================================================\n")

    TOTAL_ROUNDS = 5
    for round_i in range(1, TOTAL_ROUNDS + 1):
        print(f"\n\n{'='*20} ROUND {round_i}/{TOTAL_ROUNDS} {'='*20}")
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"\n[Round {round_i}]\n")
            
        for m_id in [1, 2, 3]:
            history = find_best_robust_features(m_id, x_raw, y_raw, feature_names)
            with open(output_filename, "a", encoding="utf-8") as f:
                f.write(f"--- Model {m_id} ---\n")
                for rec in history:
                    sel = rec['selected']
                    drop_idxs = sorted(list(all_indices - set(sel)))
                    f.write(f"  [Stress {rec['stress']}] Score: {rec['score']:.4f}\n")
                    f.write(f"    Selected: {[feature_names[i] for i in sorted(sel)]}\n")
                    f.write(f"    self.drop_indices_model{m_id} = {drop_idxs}\n\n")