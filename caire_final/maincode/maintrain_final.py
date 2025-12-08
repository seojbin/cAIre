import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, augmentdata, label
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("Preprocess/Feature Extractor module not found.")
    exit()


class HybridClassifier:
    def __init__(self, randomstate=42):
        self.randomstate = randomstate
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()
        self.scaler4 = StandardScaler()

        self.model1 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced')
        self.model2 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced')
        self.model3 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced')
        self.model4 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced')

        self.cid = label['circle']
        self.cdl = label['diagonal_left']
        self.cdr = label['diagonal_right']
        self.cho = label['horizontal']
        self.cve = label['vertical']

        # M1: Circle vs Rest (Ratio, Area, Radius, Helix, DevMax)
        self.select_indices_model1 = [9, 11, 17, 19, 20] 
        # M2: Horizontal vs Rest (Clean Range X/Y/Z, Apex Z, PCA Z)
        self.select_indices_model2 = [1, 2, 6, 7, 8, 16, 18]
        # M3: Vertical vs Diagonal (Same as M2)
        self.select_indices_model3 = [1, 2, 6, 7, 8, 16, 18]
        # M4: Diagonal L vs R (Apex X/Y, Start Rel X/Y)
        self.select_indices_model4 = [14, 15, 25, 26,28]

    def _filter_features(self, x, indices):
        return x[:, indices]

    def fit(self, x, y):
        # M1: Circle(0) vs Others(1)
        x_f1 = self._filter_features(x, self.select_indices_model1)
        self.scaler1.fit(x_f1)
        x_s1 = self.scaler1.transform(x_f1)
        y_m1 = np.where(y == self.cid, 0, 1)
        self.model1.fit(x_s1, y_m1)

        # M2: Horizontal(0) vs Others(1)
        mask_m2 = (y != self.cid)
        if np.sum(mask_m2) > 0:
            x_m2 = x[mask_m2]
            y_m2 = y[mask_m2]
            
            x_f2 = self._filter_features(x_m2, self.select_indices_model2)
            self.scaler2.fit(x_f2)
            x_s2 = self.scaler2.transform(x_f2)
            y_train2 = np.where(y_m2 == self.cho, 0, 1)
            self.model2.fit(x_s2, y_train2)

            # M3: Vertical(0) vs Diagonals(1)
            mask_m3 = (y_m2 != self.cho)
            if np.sum(mask_m3) > 0:
                x_m3 = x_m2[mask_m3]
                y_m3 = y_m2[mask_m3]
                
                x_f3 = self._filter_features(x_m3, self.select_indices_model3)
                self.scaler3.fit(x_f3)
                x_s3 = self.scaler3.transform(x_f3)
                y_train3 = np.where(y_m3 == self.cve, 0, 1)
                self.model3.fit(x_s3, y_train3)

                # M4: Left(0) vs Right(1)
                mask_m4 = (y_m3 != self.cve)
                if np.sum(mask_m4) > 0:
                    x_m4 = x_m3[mask_m4]
                    y_m4 = y_m3[mask_m4]
                    
                    x_f4 = self._filter_features(x_m4, self.select_indices_model4)
                    self.scaler4.fit(x_f4)
                    x_s4 = self.scaler4.transform(x_f4)
                    y_train4 = np.where(y_m4 == self.cdl, 0, 1)
                    self.model4.fit(x_s4, y_train4)
    def predict(self, x):
        ypred = np.zeros(len(x), dtype=int)
        
        # M1
        x_f1 = self._filter_features(x, self.select_indices_model1)
        p1 = self.model1.predict(self.scaler1.transform(x_f1))
        ypred[p1 == 0] = self.cid
        
        rest_idx = np.where(p1 == 1)[0]
        if len(rest_idx) == 0: return ypred
        
        # M2
        x_rest = x[rest_idx]
        x_f2 = self._filter_features(x_rest, self.select_indices_model2)
        p2 = self.model2.predict(self.scaler2.transform(x_f2))
        h_idx = rest_idx[p2 == 0]
        ypred[h_idx] = self.cho
        
        rest_idx2 = rest_idx[p2 == 1]
        if len(rest_idx2) == 0: return ypred
        
        # M3
        x_rest2 = x[rest_idx2]
        x_f3 = self._filter_features(x_rest2, self.select_indices_model3)
        p3 = self.model3.predict(self.scaler3.transform(x_f3))
        v_idx = rest_idx2[p3 == 0]
        ypred[v_idx] = self.cve
        
        diag_idx = rest_idx2[p3 == 1]
        if len(diag_idx) == 0: return ypred
        
        # M4
        x_diag = x[diag_idx]
        x_f4 = self._filter_features(x_diag, self.select_indices_model4)
        p4 = self.model4.predict(self.scaler4.transform(x_f4))
        
        for i, val in enumerate(p4):
            ypred[diag_idx[i]] = self.cdl if val == 0 else self.cdr
        return ypred
def visualize_model(model_clf, scaler, select_indices, x_data, y_true, title, label_map, feature_names):
    if len(x_data) == 0: return
    x_f = x_data[:, select_indices]
    x_s = scaler.transform(x_f)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    w = model_clf.coef_[0]
    b = model_clf.intercept_[0]
    eq_parts = [f"({w[i]:.2f}*{feature_names[select_indices[i]]})" for i in range(len(w))]
    print(f"\n[{title}] Hyperplane: {' + '.join(eq_parts)} + {b:.2f} = 0")

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

    preds = model_clf.predict(x_s) # Using internal model predict
    colors = ['blue', 'red']
    inv_label = {v: k for k, v in label.items()}

    for cls_label, target_val in label_map.items():
        idxs = np.where(y_true == cls_label)[0]
        if len(idxs) == 0: continue
        
        # Internal model prediction comparison
        correct_mask = (preds[idxs] == target_val)
        correct = idxs[correct_mask]
        wrong = idxs[~correct_mask]
        
        c = colors[target_val] if target_val < 2 else 'green'
        lbl = inv_label[cls_label]
        
        if len(correct) > 0:
            ax.scatter(x_vis[correct, 0], x_vis[correct, 1], x_vis[correct, 2], 
                       c=c, marker='o', s=40, alpha=0.7, label=f'{lbl}')
        if len(wrong) > 0:
            ax.scatter(x_vis[wrong, 0], x_vis[wrong, 1], x_vis[wrong, 2], 
                       c=c, marker='x', s=100, label=f'{lbl} (Wrong)')

    ax.set_title(title)
    plt.legend()
    plt.savefig(f"{title.replace(':', '').replace(' ', '_')}.png")
    plt.show() 

if __name__ == "__main__":
    # 1. Load Data (Old + New)
    data_path = os.path.join(project_root, 'data')
    newdata_path = os.path.join(project_root, 'newdata')
    
    print("Loading Data...")
    x_old, y_old = load(data_path)
    x_new, y_new = load(newdata_path)

    if len(x_new) > 0:
        x_total = x_old + x_new
        y_total = np.concatenate([y_old, y_new])
        print(f"Merged Data: {len(x_old)} (Old) + {len(x_new)} (New) = {len(x_total)} Total Samples")
    else:
        x_total = x_old
        y_total = y_old
        print(f"Loaded {len(x_total)} Samples (Old only)")

    n_aug = 9 # Total 10
    print(f"Augmenting Data x{n_aug}")
    x_aug, y_aug = augmentdata(x_total, y_total, n=n_aug)
    
    x_feat_train, feature_names = extractfeatures(x_aug)

    model = HybridClassifier()
    model.fit(x_feat_train, y_aug)
    
    joblib.dump(model, 'mainmodel_final.joblib')
    print("Model saved to 'mainmodel_final.joblib'")
    
    try:
        visualize_model(model.model1, model.scaler1, model.select_indices_model1, x_feat_train, y_aug, 
                        "Final M1: Circle vs Rest",
                        {label['circle']: 0, label['horizontal']: 1, label['vertical']: 1, 
                         label['diagonal_left']: 1, label['diagonal_right']: 1}, feature_names)
        
        m2_mask = (y_aug != label['circle'])
        if np.sum(m2_mask) > 0:
            visualize_model(model.model2, model.scaler2, model.select_indices_model2, 
                            x_feat_train[m2_mask], y_aug[m2_mask], 
                            "Final M2: Horizontal vs Rest",
                            {label['horizontal']: 0, label['vertical']: 1, 
                             label['diagonal_left']: 1, label['diagonal_right']: 1}, feature_names)
            
        m3_mask = m2_mask & (y_aug != label['horizontal'])
        if np.sum(m3_mask) > 0:
            visualize_model(model.model3, model.scaler3, model.select_indices_model3, 
                            x_feat_train[m3_mask], y_aug[m3_mask], 
                            "Final M3: Vertical vs Diagonal",
                            {label['vertical']: 0, label['diagonal_left']: 1, 
                             label['diagonal_right']: 1}, feature_names)

        m4_mask = m3_mask & (y_aug != label['vertical'])
        if np.sum(m4_mask) > 0:
            visualize_model(model.model4, model.scaler4, model.select_indices_model4, 
                            x_feat_train[m4_mask], y_aug[m4_mask], 
                            "Final M4: Diag L vs R",
                            {label['diagonal_left']: 0, label['diagonal_right']: 1}, feature_names)
    except Exception as e:
        print(f"Visualization Skipped: {e}")