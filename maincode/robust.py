import numpy as np
import pandas as pd
import sys
import os
import contextlib
import copy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.transform import Rotation as R

# =========================================================
#  CONFIGURATION
# =========================================================

# 테스트할 노이즈 모드 선택
# 옵션: 'GAUSSIAN', 'SPIKE', 'DRIFT', 'ROTATION', 'BIAS', 'LINEAR_DRIFT', 'LINEAR_DISTORTION'
NOISE_MODE = 'LINEAR_DRIFT' 

if NOISE_MODE == 'GAUSSIAN':
    STRESS_LEVELS = [10.0, 30.0, 50.0]
elif NOISE_MODE == 'SPIKE':
    STRESS_LEVELS = [100.0, 500.0, 1000.0, 2000.0]
elif NOISE_MODE == 'DRIFT':
    STRESS_LEVELS = [1.0, 3.0, 5.0]
elif NOISE_MODE == 'ROTATION':
    STRESS_LEVELS = [5.0, 15.0, 30.0, 40.0]
elif NOISE_MODE == 'BIAS':
    STRESS_LEVELS = [20.0, 50.0, 100.0, 200.0]
elif NOISE_MODE == 'LINEAR_DRIFT':
    # 기존: 직선형 이탈
    STRESS_LEVELS = [0.5, 1.0, 2.0]
elif NOISE_MODE == 'LINEAR_DISTORTION':
    # [NEW] 곡선형 왜곡 (활처럼 휨)
    # 데이터 스케일(500) 대비 약 2%(10) ~ 10%(50) 수준의 왜곡
    STRESS_LEVELS = [100.0, 200.0, 300.0]

VAL_COPY_N = 10
TRAIN_AUG_N = 9
ITERATIONS = 5

FEATURE_CONFIGS = {
    "Baseline": {
        "M1": [5, 9, 11, 13, 17, 19, 20],
        "M2": [1, 2, 6, 7, 8, 18],
        "M3": [1, 2, 6, 7, 8, 18],
        "M4": [14, 15]
    },
    "Experimental_V1": {
        "M1": [5, 9, 11, 13, 17, 19, 20],
        "M2": [1, 2, 6, 7, 8, 16, 18],
        "M3": [1, 2, 6, 7, 8, 16, 18],
        "M4": [13, 14, 15]
    },
}

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label, augmentdata, remove_spikes, smooth_trajectory
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("Error: Postprocess package not found.")
    exit()

inv_label = {v: k for k, v in label.items()}

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old

def custom_noise_augment(x_raw, y_raw, n_copies, level, mode):
    aug_x = []
    aug_y = []
    for traj, lbl in zip(x_raw, y_raw):
        aug_x.append(traj)
        aug_y.append(lbl)
        for _ in range(n_copies):
            new_traj = traj.copy()
            
            if mode == 'GAUSSIAN':
                noise = np.random.normal(loc=0.0, scale=level, size=traj.shape)
                new_traj += noise

            elif mode == 'SPIKE':
                n_points = len(traj)
                n_spikes = np.random.randint(1, 4)
                spike_indices = np.random.choice(n_points, n_spikes, replace=False)
                for idx in spike_indices:
                    direction = np.random.randn(3)
                    new_traj[idx] += direction * level

            elif mode == 'DRIFT':
                # Random Walk
                drift_step = np.random.normal(loc=0.0, scale=level, size=traj.shape)
                drift = np.cumsum(drift_step, axis=0)
                new_traj += drift

            elif mode == 'ROTATION':
                angle_rad = np.radians(np.random.uniform(-level, level))
                c, s = np.cos(angle_rad), np.sin(angle_rad)
                R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                new_traj = np.dot(new_traj, R_z.T)

            elif mode == 'BIAS':
                offset = np.random.normal(loc=0.0, scale=level, size=(1, 3))
                new_traj += offset

            elif mode == 'LINEAR_DRIFT':
                # Straight Line Drift (선형 이탈)
                drift_dir = np.random.normal(size=(1, 3))
                drift_dir /= (np.linalg.norm(drift_dir) + 1e-6)
                steps = np.arange(len(traj)).reshape(-1, 1)
                directional_error = steps * (drift_dir * level)
                new_traj += directional_error

            elif mode == 'LINEAR_DISTORTION':
                # [NEW] Curvilinear Distortion (활처럼 휘는 왜곡)
                n_points = len(traj)
                drift_dir = np.random.normal(size=(1, 3))
                drift_dir /= (np.linalg.norm(drift_dir) + 1e-6)
                
                # 0~1 사이로 정규화된 step 생성
                steps = np.linspace(0, 1, n_points).reshape(-1, 1)
                
                # [핵심] 제곱을 통해 시작점은 유지하고 끝부분으로 갈수록 휘어지게 만듦 (Quadratic)
                steps = steps ** 2 
                
                # level은 끝점이 이동할 최대 거리 (예: 30.0)
                distortion = steps * (drift_dir * level)
                new_traj += distortion
                
            aug_x.append(new_traj)
            aug_y.append(lbl)
            
    return aug_x, np.array(aug_y)

class HybridClassifierRobustness:
    def __init__(self, feature_config, feature_names_list, kernel='linear', config_name="Unknown", randomstate=42):
        self.randomstate = randomstate
        self.feature_names_list = feature_names_list
        self.total_feature_count = len(feature_names_list)
        self.config_name = config_name
        self.kernel = kernel

        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        self.scaler4 = StandardScaler()

        self.model1 = SVC(kernel=kernel, random_state=randomstate, class_weight='balanced')
        self.model2 = SVC(kernel=kernel, random_state=randomstate, class_weight='balanced')
        self.model3 = SVC(kernel=kernel, random_state=randomstate, class_weight='balanced')
        self.model4 = SVC(kernel=kernel, random_state=randomstate, class_weight='balanced')

        self.cid = label['circle']
        self.cdl = label['diagonal_left']
        self.cdr = label['diagonal_right']
        self.cho = label['horizontal']
        self.cve = label['vertical']

        self.select_indices_model1 = feature_config["M1"]
        self.select_indices_model2 = feature_config["M2"]
        self.select_indices_model3 = feature_config["M3"]
        self.select_indices_model4 = feature_config["M4"]

    def _filter_features(self, x, indices):
        return x[:, indices]

    def fit(self, x, y):
        # M1
        x_f1 = self._filter_features(x, self.select_indices_model1)
        self.scaler1.fit(x_f1)
        self.model1.fit(self.scaler1.transform(x_f1), np.where(y == self.cid, 0, 1))
        # M2
        mask_m2 = (y != self.cid)
        x_m2, y_m2 = x[mask_m2], y[mask_m2]
        if len(x_m2) > 0:
            x_f2 = self._filter_features(x_m2, self.select_indices_model2)
            self.scaler2.fit(x_f2)
            self.model2.fit(self.scaler2.transform(x_f2), np.where(y_m2 == self.cho, 0, 1))
            # M3
            mask_m3 = (y_m2 != self.cho)
            x_m3, y_m3 = x_m2[mask_m3], y_m2[mask_m3]
            if len(x_m3) > 0:
                x_f3 = self._filter_features(x_m3, self.select_indices_model3)
                self.scaler3.fit(x_f3)
                self.model3.fit(self.scaler3.transform(x_f3), np.where(y_m3 == self.cve, 0, 1))
                # M4
                mask_m4 = (y_m3 != self.cve)
                x_m4, y_m4 = x_m3[mask_m4], y_m3[mask_m4]
                if len(x_m4) > 0:
                    x_f4 = self._filter_features(x_m4, self.select_indices_model4)
                    self.scaler4.fit(x_f4)
                    self.model4.fit(self.scaler4.transform(x_f4), np.where(y_m4 == self.cdl, 0, 1))

    def predict(self, x):
        ypred = np.zeros(len(x), dtype=int)
        
        x_f1 = self._filter_features(x, self.select_indices_model1)
        p1 = self.model1.predict(self.scaler1.transform(x_f1))
        ypred[p1 == 0] = self.cid
        
        rest_idx = np.where(p1 == 1)[0]
        if len(rest_idx) == 0: return ypred

        x_rest = x[rest_idx]
        x_f2 = self._filter_features(x_rest, self.select_indices_model2)
        p2 = self.model2.predict(self.scaler2.transform(x_f2))
        
        h_idx = rest_idx[p2 == 0]
        ypred[h_idx] = self.cho
        
        rest_idx2 = rest_idx[p2 == 1]
        if len(rest_idx2) == 0: return ypred

        x_rest2 = x[rest_idx2]
        x_f3 = self._filter_features(x_rest2, self.select_indices_model3)
        p3 = self.model3.predict(self.scaler3.transform(x_f3))
        
        v_idx = rest_idx2[p3 == 0]
        ypred[v_idx] = self.cve
        
        diag_idx = rest_idx2[p3 == 1]
        if len(diag_idx) == 0: return ypred

        x_diag = x[diag_idx]
        x_f4 = self._filter_features(x_diag, self.select_indices_model4)
        p4 = self.model4.predict(self.scaler4.transform(x_f4))
        
        for i, val in enumerate(p4):
            ypred[diag_idx[i]] = self.cdl if val == 0 else self.cdr
        return ypred

    def diagnose(self, x_single, y_true):
        x = x_single.reshape(1, -1)
        # M1 Check
        p1 = self.model1.predict(self.scaler1.transform(self._filter_features(x, self.select_indices_model1)))[0]
        if y_true == self.cid: return ("M1", "M1 (Missed Circle)") if p1 != 0 else (None, None)
        if p1 == 0: return ("M1", "M1 (False Circle)")
        # M2 Check
        p2 = self.model2.predict(self.scaler2.transform(self._filter_features(x, self.select_indices_model2)))[0]
        if y_true == self.cho: return ("M2", "M2 (Missed Horiz)") if p2 != 0 else (None, None)
        if p2 == 0: return ("M2", "M2 (False Horiz)")
        # M3 Check
        p3 = self.model3.predict(self.scaler3.transform(self._filter_features(x, self.select_indices_model3)))[0]
        if y_true == self.cve: return ("M3", "M3 (Missed Vert)") if p3 != 0 else (None, None)
        if p3 == 0: return ("M3", "M3 (False Vert)")
        # M4 Check
        p4 = self.model4.predict(self.scaler4.transform(self._filter_features(x, self.select_indices_model4)))[0]
        target = 0 if y_true == self.cdl else 1
        if p4 != target: return ("M4", f"M4 ({'L->R' if target == 0 else 'R->L'} Fail)")
        return ("Unknown", "Unknown")

    def suggest_fix(self, x_train_full, y_train_full, x_fail_sample, y_fail_true, failed_model_name):
        suggestions = []

        if failed_model_name == "M1":
            target_indices = self.select_indices_model1
            x_train = x_train_full
            y_binary = np.where(y_train_full == self.cid, 0, 1)
            y_true_binary = 0 if y_fail_true == self.cid else 1
        elif failed_model_name == "M2":
            mask = y_train_full != self.cid
            if np.sum(mask) == 0: return "No Data"
            x_train = x_train_full[mask]
            y_binary = np.where(y_train_full[mask] == self.cho, 0, 1)
            y_true_binary = 0 if y_fail_true == self.cho else 1
            target_indices = self.select_indices_model2
        elif failed_model_name == "M3":
            mask = (y_train_full != self.cid) & (y_train_full != self.cho)
            if np.sum(mask) == 0: return "No Data"
            x_train = x_train_full[mask]
            y_binary = np.where(y_train_full[mask] == self.cve, 0, 1)
            y_true_binary = 0 if y_fail_true == self.cve else 1
            target_indices = self.select_indices_model3
        elif failed_model_name == "M4":
            mask = (y_train_full != self.cid) & (y_train_full != self.cho) & (y_train_full != self.cve)
            if np.sum(mask) == 0: return "No Data"
            x_train = x_train_full[mask]
            y_binary = np.where(y_train_full[mask] == self.cdl, 0, 1)
            y_true_binary = 0 if y_fail_true == self.cdl else 1
            target_indices = self.select_indices_model4
        else:
            return "Unknown Model"

        # Removal Test
        for idx_to_remove in target_indices:
            temp_indices = [i for i in target_indices if i != idx_to_remove]
            if not temp_indices: continue

            temp_scaler = StandardScaler()
            temp_model = SVC(kernel=self.kernel, random_state=self.randomstate, class_weight='balanced')

            x_tr_sub = x_train[:, temp_indices]
            temp_scaler.fit(x_tr_sub)
            temp_model.fit(temp_scaler.transform(x_tr_sub), y_binary)

            x_val_sub = x_fail_sample.reshape(1, -1)[:, temp_indices]
            pred = temp_model.predict(temp_scaler.transform(x_val_sub))[0]

            if pred == y_true_binary:
                feat_name = self.feature_names_list[idx_to_remove]
                suggestions.append(f"[{self.config_name}][{failed_model_name}] Remove '{feat_name}'")

        # Addition Test
        all_indices = set(range(self.total_feature_count))
        current_indices_set = set(target_indices)
        candidate_indices = list(all_indices - current_indices_set)

        for idx_to_add in candidate_indices:
            temp_indices = target_indices + [idx_to_add]

            temp_scaler = StandardScaler()
            temp_model = SVC(kernel=self.kernel, random_state=self.randomstate, class_weight='balanced')

            x_tr_sub = x_train[:, temp_indices]
            temp_scaler.fit(x_tr_sub)
            temp_model.fit(temp_scaler.transform(x_tr_sub), y_binary)

            x_val_sub = x_fail_sample.reshape(1, -1)[:, temp_indices]
            pred = temp_model.predict(temp_scaler.transform(x_val_sub))[0]

            if pred == y_true_binary:
                feat_name = self.feature_names_list[idx_to_add]
                suggestions.append(f"[{self.config_name}][{failed_model_name}] Add '{feat_name}'")

        if not suggestions:
            return "No simple fix found"

        return ", ".join(list(set(suggestions))[:3])


def main():
    data_path = os.path.join(project_root, 'data')
    newdata_path = os.path.join(project_root, 'newdata')
    test_size_n = 20

    with suppress_stdout():
        x_old, y_old = load(data_path)
        x_new, y_new = load(newdata_path)

    if len(x_new) > 0:
        x_total_raw = x_old + x_new
        y_total = np.concatenate([y_old, y_new])
        test_size_n = len(x_new)
        print(f"Validation Size Set to: {test_size_n} (Matched to NewData size)")
    else:
        x_total_raw = x_old
        y_total = y_old

    print(f"\nRobustness Test: {NOISE_MODE} Mode")
    print(f" -> Levels: {STRESS_LEVELS}")
    print(f" -> Configs to Test: {list(FEATURE_CONFIGS.keys())}")

    sss = StratifiedShuffleSplit(n_splits=ITERATIONS, test_size=test_size_n, random_state=42)
    error_logs = []

    with suppress_stdout():
        _, all_feature_names = extractfeatures([x_total_raw[0]])

    for i, (train_idx, val_idx) in enumerate(sss.split(x_total_raw, y_total)):

        if i == 0:
            if len(x_new) == 0: continue
            print(f"\n[Iter 1] Special Mode: Train(data) vs Val(newdata)")
            x_train_raw = x_old
            y_train = y_old
            val_real_indices = list(range(len(x_old), len(x_total_raw)))
        else:
            x_train_raw = [x_total_raw[k] for k in train_idx]
            y_train = y_total[train_idx]
            val_real_indices = val_idx

        with suppress_stdout():
            x_aug, y_aug = augmentdata(x_train_raw, y_train, n=TRAIN_AUG_N)
            x_feat_train, _ = extractfeatures(x_aug)

        # Loop through Kernels
        for kernel_type in ['linear']:
            for conf_name, conf_data in FEATURE_CONFIGS.items():
                model = HybridClassifierRobustness(
                    feature_config=conf_data,
                    feature_names_list=all_feature_names,
                    kernel=kernel_type,
                    config_name=conf_name,
                    randomstate=42
                )
                model.fit(x_feat_train, y_aug)

                for level in STRESS_LEVELS:
                    fails = 0
                    total_tests = 0

                    for idx in val_real_indices:
                        # Noise Injection
                        x_val_noisy, y_val = custom_noise_augment(
                            [x_total_raw[idx]], [y_total[idx]],
                            VAL_COPY_N, level, NOISE_MODE
                        )

                        # Preprocessing
                        x_val_cleaned = []
                        for traj in x_val_noisy:
                            cleaned_traj = remove_spikes(traj, threshold_std=5.0)
                            cleaned_traj = smooth_trajectory(cleaned_traj, window_size=3)
                            x_val_cleaned.append(cleaned_traj)

                        # Prediction
                        with suppress_stdout():
                            x_v_feat, _ = extractfeatures(x_val_cleaned)

                        preds = model.predict(x_v_feat)
                        total_tests += len(preds)

                        if np.any(preds != y_val):
                            fails += len(np.where(preds != y_val)[0])

                            err_idx_in_batch = np.where(preds != y_val)[0][0]
                            pred_label_code = preds[err_idx_in_batch]
                            true_label_code = y_val[err_idx_in_batch]

                            model_name, cause = model.diagnose(x_v_feat[err_idx_in_batch], true_label_code)

                            # Determine active features for the failed model
                            failed_model_features = "Unknown"
                            if model_name == "M1":
                                feat_indices = conf_data["M1"]
                                failed_model_features = [all_feature_names[fi] for fi in feat_indices]
                            elif model_name == "M2":
                                feat_indices = conf_data["M2"]
                                failed_model_features = [all_feature_names[fi] for fi in feat_indices]
                            elif model_name == "M3":
                                feat_indices = conf_data["M3"]
                                failed_model_features = [all_feature_names[fi] for fi in feat_indices]
                            elif model_name == "M4":
                                feat_indices = conf_data["M4"]
                                failed_model_features = [all_feature_names[fi] for fi in feat_indices]

                            suggestion = "N/A"
                            if model_name != "Unknown":
                                suggestion = model.suggest_fix(
                                    x_feat_train, y_aug,
                                    x_v_feat[err_idx_in_batch], true_label_code,
                                    model_name
                                )

                            true_lbl_str = inv_label[true_label_code]
                            pred_lbl_str = inv_label[pred_label_code]

                            error_logs.append({
                                'Iter': i + 1,
                                'Config': conf_name,
                                'Kernel': kernel_type,
                                'Mode': NOISE_MODE,
                                'Level': level,
                                'Sample_ID': idx,
                                'Label': true_lbl_str,
                                'Predicted': pred_lbl_str,
                                'Cause': cause,
                                'Failed_Model': model_name,
                                'Active_Features': str(failed_model_features),
                                'Suggestion': suggestion
                            })

                    acc = 1.0 - (fails / total_tests)
                    print(f"   Iter {i + 1} | {conf_name:<15} ({kernel_type}) | Lvl {level}: Acc {acc * 100:.1f}%")

    with open("robustreport.txt", "w", encoding="utf-8") as f:
        f.write("Robustness Report\n")
        f.write("=================\n\n")

        f.write("[Feature Definitions by Configuration]\n")
        for c_name, c_cfg in FEATURE_CONFIGS.items():
            f.write(f"Configuration: {c_name}\n")
            for m_key, m_indices in c_cfg.items():
                m_feat_names = [all_feature_names[i] for i in m_indices]
                f.write(f"  - {m_key}: {m_feat_names}\n")
            f.write("\n")
        f.write("-" * 50 + "\n\n")

        if not error_logs:
            f.write("PERFECT SCORE across all configs and levels!\n")
            print("\nAll configurations operated successfully.")
        else:
            df = pd.DataFrame(error_logs)

            f.write("[Failure Counts by Config, Kernel & Level]\n")
            # Group by Config AND Kernel
            summary = df.groupby(['Config', 'Kernel', 'Level']).size().unstack(fill_value=0)
            f.write(summary.to_string())
            f.write("\n\n")

            f.write("[Top Failure Causes by Config & Kernel]\n")
            for conf in df['Config'].unique():
                for kern in df['Kernel'].unique():
                    f.write(f"\n>> Configuration: {conf} ({kern})\n")
                    subset = df[(df['Config'] == conf) & (df['Kernel'] == kern)]
                    if not subset.empty:
                        f.write(subset['Cause'].value_counts().to_string())
                    else:
                        f.write("No failures.")
                    f.write("\n")
            f.write("\n")

            f.write("[Top Suggested Fixes]\n")
            valid_suggestions = df[df['Suggestion'] != "No simple fix found"]['Suggestion']
            if len(valid_suggestions) > 0:
                f.write(valid_suggestions.value_counts().head(20).to_string())
            else:
                f.write("No effective feature changes found.")
            f.write("\n\n")

            f.write("[Detailed Error Log Sample (Top 50)]\n")
            cols_to_show = ['Config', 'Kernel', 'Level', 'Label', 'Predicted', 'Cause', 'Failed_Model', 'Active_Features', 'Suggestion']
            f.write(df[cols_to_show].head(50).to_string())
            f.write("\n\n* Full details are saved in 'robust_error_details.csv'")

            df.to_csv("robust_error_details.csv", index=False, encoding='utf-8-sig')

            print(f"\nTotal failure records: {len(df)}")


if __name__ == "__main__":
    main()