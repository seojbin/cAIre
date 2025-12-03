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
    print("전처리 파일 없음")
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

        # [Feature Selection]
        # M1: Circle vs Rest (Ratio, Area, Radius Ratio)
        self.select_indices_model1 = [9, 11, 16] 

        # M2: Horizontal vs Rest (Line)
        self.select_indices_model2 = [1, 2, 6, 7, 8,17]

        # M3: Vertical vs Complex (Diagonal)
        self.select_indices_model3 = [1, 2, 6, 7, 8,17]

        # M4: Diagonal Left vs Right
        self.select_indices_model4 = [14, 15]

    def _filter_features(self, x, indices):
        return x[:, indices]

    def fit(self, x, y, xorig):
        # M1: Circle(0) vs Others(1)
        x_f1 = self._filter_features(x, self.select_indices_model1)
        self.scaler1.fit(x_f1)
        x_s1 = self.scaler1.transform(x_f1)
        y_m1 = np.where(y == self.cid, 0, 1)
        self.model1.fit(x_s1, y_m1)

        # M2: Horizontal(0) vs Others(1) (Non-Circle Only)
        mask_m2 = (y != self.cid)
        if np.sum(mask_m2) > 0:
            x_m2 = x[mask_m2]
            y_m2 = y[mask_m2]
            
            x_f2 = self._filter_features(x_m2, self.select_indices_model2)
            self.scaler2.fit(x_f2)
            x_s2 = self.scaler2.transform(x_f2)
            
            y_train2 = np.where(y_m2 == self.cho, 0, 1)
            self.model2.fit(x_s2, y_train2)

            # M3: Vertical(0) vs Diagonals(1) (Non-Horiz Only)
            mask_m3 = (y_m2 != self.cho)
            if np.sum(mask_m3) > 0:
                x_m3 = x_m2[mask_m3]
                y_m3 = y_m2[mask_m3]
                
                x_f3 = self._filter_features(x_m3, self.select_indices_model3)
                self.scaler3.fit(x_f3)
                x_s3 = self.scaler3.transform(x_f3)
                
                y_train3 = np.where(y_m3 == self.cve, 0, 1)
                self.model3.fit(x_s3, y_train3)

                # M4: Left(0) vs Right(1) (Diagonals Only)
                mask_m4 = (y_m3 != self.cve)
                if np.sum(mask_m4) > 0:
                    x_m4 = x_m3[mask_m4]
                    y_m4 = y_m3[mask_m4]
                    
                    x_f4 = self._filter_features(x_m4, self.select_indices_model4)
                    self.scaler4.fit(x_f4)
                    x_s4 = self.scaler4.transform(x_f4)
                    
                    y_train4 = np.where(y_m4 == self.cdl, 0, 1)
                    self.model4.fit(x_s4, y_train4)

    def predict(self, x, xorig):
        ypred = np.zeros(len(x), dtype=int)

        # M1: Circle
        x_f1 = self._filter_features(x, self.select_indices_model1)
        p1 = self.model1.predict(self.scaler1.transform(x_f1))
        ypred[p1 == 0] = self.cid
        
        rest_idx = np.where(p1 == 1)[0]
        if len(rest_idx) == 0: return ypred

        # M2: Horizontal
        x_rest = x[rest_idx]
        x_f2 = self._filter_features(x_rest, self.select_indices_model2)
        p2 = self.model2.predict(self.scaler2.transform(x_f2))
        
        h_idx = rest_idx[p2 == 0]
        ypred[h_idx] = self.cho
        
        rest_idx2 = rest_idx[p2 == 1]
        if len(rest_idx2) == 0: return ypred

        # M3: Vertical
        x_rest2 = x[rest_idx2]
        x_f3 = self._filter_features(x_rest2, self.select_indices_model3)
        p3 = self.model3.predict(self.scaler3.transform(x_f3))
        
        v_idx = rest_idx2[p3 == 0]
        ypred[v_idx] = self.cve
        
        diag_idx = rest_idx2[p3 == 1]
        if len(diag_idx) == 0: return ypred

        # M4: Diagonal L vs R
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
        
        x_min, x_max = x_vis[:, 0].min() - 1, x_vis[:, 0].max() + 1
        y_min, y_max = x_vis[:, 1].min() - 1, x_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
        
        if abs(w_pca[2]) > 0.001:
            z = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
            ax.plot_surface(xx, yy, z, alpha=0.2, color='gray')
    else:
        x_vis = np.hstack([x_s, np.zeros((len(x_s), 3 - x_s.shape[1]))])
        x_min, x_max = x_vis[:, 0].min() - 1, x_vis[:, 0].max() + 1
        if abs(w[1]) > 0.001:
            zz, xx_v = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(x_min, x_max, 10))
            yy_v = -(w[0] * xx_v + b) / w[1]
            ax.plot_surface(xx_v, yy_v, zz, alpha=0.2, color='gray')

    preds = model_clf.predict(x_s)
    colors = ['blue', 'red']
    inv_label = {v: k for k, v in label.items()}

    for cls_label, target_val in label_map.items():
        idxs = np.where(y_true == cls_label)[0]
        if len(idxs) == 0: continue
        
        correct = idxs[preds[idxs] == target_val]
        wrong = idxs[preds[idxs] != target_val]
        
        c = colors[target_val] if target_val < 2 else 'green'
        lbl = inv_label[cls_label]
        
        if len(correct) > 0:
            ax.scatter(x_vis[correct, 0], x_vis[correct, 1], x_vis[correct, 2], 
                       c=c, marker='o', s=40, alpha=0.7, label=f'{lbl}')
        if len(wrong) > 0:
            ax.scatter(x_vis[wrong, 0], x_vis[wrong, 1], x_vis[wrong, 2], 
                       c=c, marker='x', s=100, label=f'{lbl} (Wrong)')

    ax.set_title(title)
    ax.view_init(elev=25, azim=135)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = os.path.join(project_root, 'data')
    naugment = 9
    xorig, yorig = load(data)

    xtrainorig, xtestorig, ytrainorig, ytest = train_test_split(xorig, yorig, test_size=0.2, stratify=yorig, random_state=42)
    xauglist, ytrainaug = augmentdata(xtrainorig, ytrainorig, n=naugment)
    xtrain, feature_names = extractfeatures(xauglist) # [수정] featurenames -> feature_names
    xtest, _ = extractfeatures(xtestorig)

    model = HybridClassifier()
    model.fit(xtrain, ytrainaug, xauglist)
    joblib.dump(model, 'mainmodel.joblib')

    # 평가
    ypred = model.predict(xtest, xtestorig)
    
    classnames = list(label.keys())
    predicted_labels = [classnames[p] for p in ypred]
    true_labels = [classnames[t] for t in ytest]
    
    print("\n" + "="*30)
    print(f"Test Set Evaluation ({len(xtest)} samples)")
    for i in range(len(ypred)):
        print(f"Sample {i+1}: Predict={predicted_labels[i]}, True={true_labels[i]}")
    
    print("\n" + "="*30)
    print(classification_report(ytest, ypred, target_names=classnames, zero_division=0))
    
    cm = confusion_matrix(ytest, ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classnames, yticklabels=classnames)
    plt.title('Confusion Matrix')
    plt.show()

    # 시각화
    try:
        # [수정] featurenames -> feature_names
        visualize_model(model.model1, model.scaler1, model.select_indices_model1, xtrain, ytrainaug, 
                        "M1: Circle vs Rest (Ratio, Area, Radius)",
                        {label['circle']: 0, label['horizontal']: 1, label['vertical']: 1, 
                         label['diagonal_left']: 1, label['diagonal_right']: 1}, feature_names)
        
        m2_mask = (ytrainaug != label['circle'])
        if np.sum(m2_mask) > 0:
            visualize_model(model.model2, model.scaler2, model.select_indices_model2, 
                            xtrain[m2_mask], ytrainaug[m2_mask], 
                            "M2: Horizontal vs Rest",
                            {label['horizontal']: 0, label['vertical']: 1, 
                             label['diagonal_left']: 1, label['diagonal_right']: 1}, feature_names)
            
        m3_mask = m2_mask & (ytrainaug != label['horizontal'])
        if np.sum(m3_mask) > 0:
            visualize_model(model.model3, model.scaler3, model.select_indices_model3, 
                            xtrain[m3_mask], ytrainaug[m3_mask], 
                            "M3: Vertical vs Diagonal",
                            {label['vertical']: 0, label['diagonal_left']: 1, 
                             label['diagonal_right']: 1}, feature_names)

        m4_mask = m3_mask & (ytrainaug != label['vertical'])
        if np.sum(m4_mask) > 0:
            visualize_model(model.model4, model.scaler4, model.select_indices_model4, 
                            xtrain[m4_mask], ytrainaug[m4_mask], 
                            "M4: Diag L vs R (Apex)",
                            {label['diagonal_left']: 0, label['diagonal_right']: 1}, feature_names)
            
    except Exception as e:
        print(f"Visualization Error: {e}")