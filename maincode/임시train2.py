import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
    exit()

class HybridClassifier:
    def __init__(self, randomstate=42):
        self.randomstate = randomstate
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()

        self.model1 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced') # H vs Rest
        self.model2 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced') # V vs Complex
        self.model3 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced') # C vs D
        self.model4 = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric='dtw') # L vs R

        self.cid = label['circle']
        self.cdl = label['diagonal_left']
        self.cdr = label['diagonal_right']
        self.cho = label['horizontal']
        self.cve = label['vertical']

        # Default or Optimized Indices
        self.drop_indices_model1 = [3, 4, 5, 9, 10, 12, 13] 
        self.drop_indices_model2 = [3, 4, 5, 9, 10, 12, 13] 
        self.drop_indices_model3 = [] 

    def _filter_features(self, x, indices):
        return np.delete(x, indices, axis=1)

    def fit(self, x, y, xorig):
        # 1. H vs Rest
        x_f1 = self._filter_features(x, self.drop_indices_model1)
        self.scaler1.fit(x_f1)
        x_s1 = self.scaler1.transform(x_f1)
        y_m1 = np.where(y == self.cho, 0, 1)
        self.model1.fit(x_s1, y_m1)

        # 2. V vs Complex
        mask_m2 = (y != self.cho)
        x_m2 = x[mask_m2]
        y_m2_orig = y[mask_m2]
        if len(x_m2) > 0:
            x_f2 = self._filter_features(x_m2, self.drop_indices_model2)
            self.scaler2.fit(x_f2)
            x_s2 = self.scaler2.transform(x_f2)
            y_m2 = np.where(y_m2_orig == self.cve, 0, 1)
            self.model2.fit(x_s2, y_m2)

            # 3. C vs D
            mask_m3 = (y_m2_orig != self.cve)
            x_m3 = x_m2[mask_m3]
            y_m3_orig = y_m2_orig[mask_m3]
            
            if len(x_m3) > 0:
                x_f3 = self._filter_features(x_m3, self.drop_indices_model3)
                self.scaler3.fit(x_f3)
                x_s3 = self.scaler3.transform(x_f3)
                y_m3 = np.where(y_m3_orig == self.cid, 0, 1)
                self.model3.fit(x_s3, y_m3)

                # 4. L vs R (KNN-DTW)
                mask_m4 = (y_m3_orig != self.cid)
                y_m4_orig = y_m3_orig[mask_m4]
                
                # Use raw time series for DTW
                # Filter xorig based on masks
                idxs_m2 = np.where(mask_m2)[0]
                idxs_m3 = idxs_m2[mask_m3]
                idxs_m4 = idxs_m3[mask_m4]
                
                xtrain4 = [xorig[i] for i in idxs_m4]
                
                if len(xtrain4) > 0:
                    xtrain4 = pad_sequences(xtrain4, padding='post', dtype='float32', value=np.nan)
                    self.model4.fit(xtrain4, y_m4_orig)

    def predict(self, x, xorig):
        ypred = np.zeros(len(x), dtype=int)

        # 1. H vs Rest
        x_f1 = self._filter_features(x, self.drop_indices_model1)
        x_s1 = self.scaler1.transform(x_f1)
        p1 = self.model1.predict(x_s1)

        h_idx = np.where(p1 == 0)[0]
        ypred[h_idx] = self.cho
        
        rest_idx = np.where(p1 == 1)[0]
        if len(rest_idx) == 0: return ypred

        # 2. V vs Complex
        x_rest = x[rest_idx]
        x_f2 = self._filter_features(x_rest, self.drop_indices_model2)
        x_s2 = self.scaler2.transform(x_f2)
        p2 = self.model2.predict(x_s2)

        v_local_idx = np.where(p2 == 0)[0]
        ypred[rest_idx[v_local_idx]] = self.cve
        
        comp_local_idx = np.where(p2 == 1)[0]
        if len(comp_local_idx) == 0: return ypred
        
        comp_global_idx = rest_idx[comp_local_idx]

        # 3. C vs D
        x_comp = x[comp_global_idx]
        x_f3 = self._filter_features(x_comp, self.drop_indices_model3)
        x_s3 = self.scaler3.transform(x_f3)
        p3 = self.model3.predict(x_s3)

        c_local_idx = np.where(p3 == 0)[0]
        ypred[comp_global_idx[c_local_idx]] = self.cid
        
        diag_local_idx = np.where(p3 == 1)[0]
        if len(diag_local_idx) == 0: return ypred
        
        diag_global_idx = comp_global_idx[diag_local_idx]

        # 4. L vs R (KNN)
        xtest4 = [xorig[i] for i in diag_global_idx]
        xtest4 = pad_sequences(xtest4, padding='post', dtype='float32', value=np.nan)
        p4 = self.model4.predict(xtest4)

        ypred[diag_global_idx] = p4

        return ypred

def visualize_binary(model, model_obj, scaler, drop_indices, x_data, y_true, title, label_map, feature_names):
    x_filtered = model._filter_features(x_data, drop_indices)
    x_scaled = scaler.transform(x_filtered)
    
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x_scaled)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    w = model_obj.coef_[0]
    b = model_obj.intercept_[0]
    
    # Equation
    filtered_names = [n for i, n in enumerate(feature_names) if i not in drop_indices]
    eq_str = " + ".join([f"{w[i]:.3f}*{filtered_names[i]}" for i in range(len(w))])
    print(f"\n[{title}] Hyperplane Equation:\n {eq_str} + {b:.3f} = 0")

    w_pca = w @ pca.components_.T
    b_pca = b + np.dot(w, pca.mean_)
    
    x_min, x_max = x_pca[:, 0].min()-1, x_pca[:, 0].max()+1
    y_min, y_max = x_pca[:, 1].min()-1, x_pca[:, 1].max()+1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    if abs(w_pca[2]) > 0.001:
        z = -(w_pca[0]*xx + w_pca[1]*yy + b_pca) / w_pca[2]
        ax.plot_surface(xx, yy, z, alpha=0.2, color='gray')
    
    colors = ['blue', 'red']
    preds = model_obj.predict(x_scaled)
    
    for cls, binary_target in label_map.items():
        idxs = np.where(y_true == cls)[0]
        if len(idxs) == 0: continue
        
        correct = idxs[preds[idxs] == binary_target]
        wrong = idxs[preds[idxs] != binary_target]
        
        c = colors[binary_target]
        ax.scatter(x_pca[correct,0], x_pca[correct,1], x_pca[correct,2], c=c, marker='o', s=40, alpha=0.8, label=f'Class {cls}')
        ax.scatter(x_pca[wrong,0], x_pca[wrong,1], x_pca[wrong,2], c=c, marker='x', s=100)
        
    ax.set_title(title)
    plt.legend()
    plt.show()

data = os.path.join(project_root, 'data')
naugment = 9
xorig, yorig = load(data)
xtrainorig, xtestorig, ytrainorig, ytest = train_test_split(xorig, yorig, test_size=0.2, stratify=yorig, random_state=42)
xauglist, ytrainaug = augmentdata(xtrainorig, ytrainorig, n=naugment)
xtrain, featurenames = extractfeatures(xauglist)
xtest, _ = extractfeatures(xtestorig)

model = HybridClassifier()
model.fit(xtrain, ytrainaug, xauglist)
joblib.dump(model, 'mainmodel.joblib')

ypred = model.predict(xtest, xtestorig)
print(classification_report(ytest, ypred, target_names=list(label.keys()), zero_division=0))

# Visualizations
visualize_binary(model, model.model1, model.scaler1, model.drop_indices_model1, xtrain, ytrainaug, 
                 "Model 1: Horiz vs Rest", 
                 {label['horizontal']: 0, label['vertical']: 1, label['circle']: 1, label['diagonal_left']: 1, label['diagonal_right']: 1},
                 featurenames)

visualize_binary(model, model.model2, model.scaler2, model.drop_indices_model2, xtrain[ytrainaug != label['horizontal']], ytrainaug[ytrainaug != label['horizontal']], 
                 "Model 2: Vert vs Complex", 
                 {label['vertical']: 0, label['circle']: 1, label['diagonal_left']: 1, label['diagonal_right']: 1},
                 featurenames)

visualize_binary(model, model.model3, model.scaler3, model.drop_indices_model3, 
                 xtrain[(ytrainaug != label['horizontal']) & (ytrainaug != label['vertical'])], 
                 ytrainaug[(ytrainaug != label['horizontal']) & (ytrainaug != label['vertical'])], 
                 "Model 3: Circle vs Diagonal", 
                 {label['circle']: 0, label['diagonal_left']: 1, label['diagonal_right']: 1},
                 featurenames)