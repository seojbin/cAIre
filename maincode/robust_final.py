import numpy as np
import pandas as pd
import sys
import os
import contextlib
import datetime
import importlib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# CONFIGURATION
EXTRACTOR_MODULE_NAME = 'feature_extractor' 

NOISE_SCENARIOS = {
    'DRIFT': [2.0, 4.0, 6.0],
    'LINEAR_DRIFT': [1.0, 3.0, 5.0],
    'ROTATION': [15.0, 30.0, 40.0],
    'LINEAR_DISTORTION': [200.0, 400.0, 600.0],
    'GAUSSIAN': [10.0, 30.0, 50.0],
    'SPIKE': [500.0, 1000.0, 2000.0],
    'BIAS': [20.0, 50.0, 100.0]
}

VAL_COPY_N = 5
TRAIN_AUG_N = 9
ITERATIONS = 1

FEATURE_CONFIG = {
    "M1": [9, 11, 17, 19, 20],
    "M2": [1, 2, 6, 7, 8, 16, 18],
    "M3": [1, 2, 6, 7, 8, 16, 18],
    "M4": [14, 15, 25, 26]
}

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label, augmentdata, remove_spikes, smooth_trajectory
    extractor_module = importlib.import_module(f"postprocess.{EXTRACTOR_MODULE_NAME}")
    extractfeatures = extractor_module.extractfeatures
except ImportError as e:
    print(f"Error: {e}")
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

def apply_single_noise(traj, mode, level):
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
        drift_dir = np.random.normal(size=(1, 3))
        drift_dir /= (np.linalg.norm(drift_dir) + 1e-6)
        steps = np.arange(len(traj)).reshape(-1, 1)
        directional_error = steps * (drift_dir * level)
        new_traj += directional_error
    elif mode == 'LINEAR_DISTORTION':
        n_points = len(traj)
        drift_dir = np.random.normal(size=(1, 3))
        drift_dir /= (np.linalg.norm(drift_dir) + 1e-6)
        steps = np.linspace(0, 1, n_points).reshape(-1, 1)
        steps = steps ** 2 
        distortion = steps * (drift_dir * level)
        new_traj += distortion
        
    return new_traj

def custom_noise_augment(x_raw, y_raw, n_copies, level, mode_input):
    aug_x = []
    aug_y = []
    aug_raw_pairs = []
    modes = mode_input if isinstance(mode_input, list) else [mode_input]

    for traj, lbl in zip(x_raw, y_raw):
        aug_x.append(traj)
        aug_y.append(lbl)
        aug_raw_pairs.append((traj, traj)) 
        
        for _ in range(n_copies):
            temp_traj = traj.copy()
            for m in modes:
                temp_traj = apply_single_noise(temp_traj, m, level)
            aug_x.append(temp_traj)
            aug_y.append(lbl)
            aug_raw_pairs.append((traj, temp_traj))
            
    return aug_x, np.array(aug_y), aug_raw_pairs

def save_trajectory_vis(pairs, labels, mode, level, save_dir, sample_n=3):
    fig = plt.figure(figsize=(15, 5))
    indices = np.random.choice(len(pairs), min(len(pairs), sample_n), replace=False)
    
    for i, idx in enumerate(indices):
        orig, noisy = pairs[idx]
        lbl_name = inv_label[labels[idx]]
        
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot(orig[:,0], orig[:,1], orig[:,2], 'b--', label='Original', alpha=0.3, linewidth=1)
        ax.scatter(orig[0,0], orig[0,1], orig[0,2], c='green', marker='o', s=20)
        
        ax.plot(noisy[:,0], noisy[:,1], noisy[:,2], 'r-', label=f'{mode} Lvl{level}', alpha=0.15, linewidth=1)
        ax.scatter(noisy[:,0], noisy[:,1], noisy[:,2], c='red', marker='.', s=10, alpha=0.6)
        ax.scatter(noisy[-1,0], noisy[-1,1], noisy[-1,2], c='black', marker='x', s=30) 
        
        ax.set_title(f"{lbl_name}")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Traj_Viz_{mode}_L{level}.png"))
    plt.close()

class HybridClassifierRobustness:
    def __init__(self, feature_config, feature_names_list, kernel='linear', randomstate=42):
        self.randomstate = randomstate
        self.feature_names_list = feature_names_list
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

        self.select_indices_model1 = self._resolve_indices(feature_config["M1"])
        self.select_indices_model2 = self._resolve_indices(feature_config["M2"])
        self.select_indices_model3 = self._resolve_indices(feature_config["M3"])
        self.select_indices_model4 = self._resolve_indices(feature_config["M4"])

    def _resolve_indices(self, config_item):
        indices = []
        for item in config_item:
            if isinstance(item, str):
                try: indices.append(self.feature_names_list.index(item))
                except: pass
            else: indices.append(item)
        return indices

    def _filter_features(self, x, indices): return x[:, indices]

    def fit(self, x, y):
        x_f1 = self._filter_features(x, self.select_indices_model1)
        self.scaler1.fit(x_f1)
        self.model1.fit(self.scaler1.transform(x_f1), np.where(y == self.cid, 0, 1))

        mask_m2 = (y != self.cid)
        if np.sum(mask_m2) > 0:
            x_m2, y_m2 = x[mask_m2], y[mask_m2]
            x_f2 = self._filter_features(x_m2, self.select_indices_model2)
            self.scaler2.fit(x_f2)
            self.model2.fit(self.scaler2.transform(x_f2), np.where(y_m2 == self.cho, 0, 1))

            mask_m3 = (y_m2 != self.cho)
            if np.sum(mask_m3) > 0:
                x_m3, y_m3 = x_m2[mask_m3], y_m2[mask_m3]
                x_f3 = self._filter_features(x_m3, self.select_indices_model3)
                self.scaler3.fit(x_f3)
                self.model3.fit(self.scaler3.transform(x_f3), np.where(y_m3 == self.cve, 0, 1))

                mask_m4 = (y_m3 != self.cve)
                if np.sum(mask_m4) > 0:
                    x_m4, y_m4 = x_m3[mask_m4], y_m3[mask_m4]
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

    def get_hyperplane_eq(self, model_clf, indices):
        if not hasattr(model_clf, 'coef_'): return "N/A"
        w = model_clf.coef_[0]
        b = model_clf.intercept_[0]
        terms = [f"({w[i]:.2f}*{self.feature_names_list[indices[i]]})" for i in range(len(w))]
        return f"{' + '.join(terms)} + {b:.2f} = 0"

    def save_model_vis(self, x_data, y_true, model_clf, scaler, indices, title, filename, label_map):
        if len(x_data) == 0: return
        x_f = x_data[:, indices]
        x_s = scaler.transform(x_f)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        w = model_clf.coef_[0]
        b = model_clf.intercept_[0]
        
        if x_f.shape[1] >= 3:
            pca = PCA(n_components=3)
            x_vis = pca.fit_transform(x_s)
            w_pca = w @ pca.components_.T
            b_pca = b + np.dot(w, pca.mean_)
            
            x_min, x_max = x_vis[:, 0].min(), x_vis[:, 0].max()
            y_min, y_max = x_vis[:, 1].min(), x_vis[:, 1].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
            if abs(w_pca[2]) > 0.001:
                z = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
                ax.plot_surface(xx, yy, z, alpha=0.2, color='gray')
        else:
            x_vis = np.hstack([x_s, np.zeros((len(x_s), 3 - x_s.shape[1]))])

        preds = model_clf.predict(x_s)
        colors = ['blue', 'red'] 
        
        for cls_lbl, target_val in label_map.items():
            idxs = np.where(y_true == cls_lbl)[0]
            if len(idxs) == 0: continue
            
            correct = idxs[preds[idxs] == target_val]
            wrong = idxs[preds[idxs] != target_val]
            
            col = colors[target_val] if target_val < 2 else 'green'
            lbl_name = inv_label[cls_lbl]
            
            if len(correct) > 0:
                ax.scatter(x_vis[correct, 0], x_vis[correct, 1], x_vis[correct, 2], 
                           c=col, marker='o', s=40, alpha=0.7, label=f'{lbl_name}')
            if len(wrong) > 0:
                ax.scatter(x_vis[wrong, 0], x_vis[wrong, 1], x_vis[wrong, 2], 
                           c='black', marker='x', s=100, label=f'{lbl_name} (Fail)')

        ax.set_title(title)
        plt.legend()
        plt.savefig(filename)
        plt.close()

def main():
    data_path = os.path.join(project_root, 'data')
    with suppress_stdout():
        x_raw, y_raw = load(data_path)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(x_raw, y_raw))
    
    x_train_raw = [x_raw[i] for i in train_idx]
    y_train = y_raw[train_idx]
    x_val_raw = [x_raw[i] for i in val_idx]
    y_val_orig = y_raw[val_idx]

    print("Augmenting Train Data...")
    x_aug, y_aug = augmentdata(x_train_raw, y_train, n=TRAIN_AUG_N)
    x_feat_train, all_feature_names = extractfeatures(x_aug)
    
    model = HybridClassifierRobustness(FEATURE_CONFIG, all_feature_names)
    model.fit(x_feat_train, y_aug)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = open(f"Robustness_Final_Report_{timestamp}.txt", "w")
    
    report_file.write(f"=== Robustness Test Report ({timestamp}) ===\n\n")
    report_file.write("[Model Hyperplanes]\n")
    report_file.write(f"M1: {model.get_hyperplane_eq(model.model1, model.select_indices_model1)}\n")
    report_file.write(f"M2: {model.get_hyperplane_eq(model.model2, model.select_indices_model2)}\n")
    report_file.write(f"M3: {model.get_hyperplane_eq(model.model3, model.select_indices_model3)}\n")
    report_file.write(f"M4: {model.get_hyperplane_eq(model.model4, model.select_indices_model4)}\n\n")

    for mode, levels in NOISE_SCENARIOS.items():
        print(f"\nTesting Mode: {mode}")
        report_file.write(f"--- Scenario: {mode} ---\n")
        
        vis_dir = f"vis_{timestamp}/{mode}"
        os.makedirs(vis_dir, exist_ok=True)

        for lvl in levels:
            x_val_noisy_flat = []
            y_val_flat = []
            aug_pairs_flat = [] 
            
            for i, traj in enumerate(x_val_raw):
                aug_trajs, aug_lbls, aug_pairs = custom_noise_augment([traj], [y_val_orig[i]], VAL_COPY_N, lvl, mode)
                for t in aug_trajs:
                    t = remove_spikes(t, threshold_std=5.0)
                    t = smooth_trajectory(t, window_size=3)
                    x_val_noisy_flat.append(t)
                y_val_flat.extend(aug_lbls)
                aug_pairs_flat.extend(aug_pairs)
                
            y_val_flat = np.array(y_val_flat)
            x_feat_val, _ = extractfeatures(x_val_noisy_flat)
            
            preds = model.predict(x_feat_val)
            acc = np.mean(preds == y_val_flat)
            
            print(f"  Level {lvl}: Accuracy {acc*100:.2f}%")
            report_file.write(f"Level {lvl}: Acc {acc*100:.2f}%\n")
            
            save_trajectory_vis(aug_pairs_flat, y_val_flat, mode, lvl, vis_dir)

            model.save_model_vis(x_feat_val, y_val_flat, model.model1, model.scaler1, 
                                 model.select_indices_model1, f"M1_{mode}_L{lvl}", 
                                 f"{vis_dir}/M1_L{lvl}.png", {label['circle']: 0, label['horizontal']: 1})
            
            mask_m2 = (y_val_flat != label['circle'])
            if np.sum(mask_m2) > 0:
                model.save_model_vis(x_feat_val[mask_m2], y_val_flat[mask_m2], model.model2, model.scaler2,
                                     model.select_indices_model2, f"M2_{mode}_L{lvl}",
                                     f"{vis_dir}/M2_L{lvl}.png", {label['horizontal']: 0, label['vertical']: 1})

            mask_m3 = mask_m2 & (y_val_flat != label['horizontal'])
            if np.sum(mask_m3) > 0:
                model.save_model_vis(x_feat_val[mask_m3], y_val_flat[mask_m3], model.model3, model.scaler3, 
                                     model.select_indices_model3, f"M3_{mode}_L{lvl}", 
                                     f"{vis_dir}/M3_L{lvl}.png", {label['vertical']: 0, label['diagonal_left']: 1})
            
            mask_m4 = mask_m3 & (y_val_flat != label['vertical'])
            if np.sum(mask_m4) > 0:
                model.save_model_vis(x_feat_val[mask_m4], y_val_flat[mask_m4], model.model4, model.scaler4,
                                     model.select_indices_model4, f"M4_{mode}_L{lvl}",
                                     f"{vis_dir}/M4_L{lvl}.png", {label['diagonal_left']: 0, label['diagonal_right']: 1})
    
    report_file.close()
    print(f"\nDone. Report saved to {report_file.name}")
    print(f"Visualizations saved to vis_{timestamp}/")

if __name__ == "__main__":
    main()