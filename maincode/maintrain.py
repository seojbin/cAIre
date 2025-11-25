import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
    print("전처리파일(preprocess.py, feature_extractor.py) 없음")
    exit()


class HybridClassifier:
    def __init__(self, maxdepthsimple=3, maxdepthcomplex=3, maxneighbor2=3, randomstate=42):
        self.randomstate = randomstate
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()

        # Model 1: H vs V vs Complex (Linear SVM)
        self.model1 = SVC(kernel='linear', random_state=self.randomstate)
        
        # Model 2: Circle vs Diagonal (Linear SVM) - 결정나무에서 변경됨
        self.model2 = SVC(kernel='linear', random_state=self.randomstate)
        
        # Model 3: Diag Left vs Diag Right (KNN-DTW) - 복구됨
        self.model3 = KNeighborsTimeSeriesClassifier(n_neighbors=maxneighbor2, metric='dtw')

        self.cid = label['circle']
        self.cdl = label['diagonal_left']
        self.cdr = label['diagonal_right']
        self.cho = label['horizontal']
        self.cve = label['vertical']
        
        self.diagonallabels = [self.cdl, self.cdr]
        self.complexlabels = [self.cid, self.cdl, self.cdr]

        # Model 1 제거 피쳐
        self.drop_indices_model1 = [3,4,5,9,10,12]
        
        # Model 2 제거 피쳐
        self.drop_indices_model2 = [3, 4, 5, 6, 8, 10]

    def _filter_features(self, x, indices):
        return np.delete(x, indices, axis=1)

    def fit(self, x, y, xorig):
        x_filtered1 = self._filter_features(x, self.drop_indices_model1)
        self.scaler1.fit(x_filtered1)
        x_scaled1 = self.scaler1.transform(x_filtered1)

        ytrain1 = np.full(y.shape, 2)
        ytrain1[y == self.cho] = 0
        ytrain1[y == self.cve] = 1
        
        self.model1.fit(x_scaled1, ytrain1)
        complexmask = (ytrain1 == 2)
        xtrain2 = x[complexmask]
        ytrain2_orig = y[complexmask]
        
        # 0: Circle, 1: Diagonal
        ytrain2 = np.where(ytrain2_orig == self.cid, 0, 1)
        
        if len(xtrain2) > 0:
            x_filtered2 = self._filter_features(xtrain2, self.drop_indices_model2)
            self.scaler2.fit(x_filtered2)
            x_scaled2 = self.scaler2.transform(x_filtered2)
            self.model2.fit(x_scaled2, ytrain2)
        diagonalmask = np.isin(y, self.diagonallabels)
        # 시계열 원본 데이터 사용
        xtrain3 = [xorig[i] for i in range(len(xorig)) if diagonalmask[i]]
        
        if len(xtrain3) > 0:
            xtrain3 = pad_sequences(xtrain3, padding='post', dtype='float32', value=np.nan)
            ytrain3 = y[diagonalmask]
            self.model3.fit(xtrain3, ytrain3)

    def predict(self, x, xorig):
        # Predict Model 1
        x_filtered1 = self._filter_features(x, self.drop_indices_model1)
        x_scaled1 = self.scaler1.transform(x_filtered1)
        ypred1 = self.model1.predict(x_scaled1)
        
        ypred = np.zeros(len(x), dtype=int)
        ypred[ypred1 == 0] = self.cho
        ypred[ypred1 == 1] = self.cve

        # Predict Complex (Circle vs Diagonal)
        complexmask = (ypred1 == 2)
        xtestcomplex = x[complexmask]
        complex_indices = np.where(complexmask)[0]
        
        if len(complex_indices) > 0:
            x_filtered2 = self._filter_features(xtestcomplex, self.drop_indices_model2)
            x_scaled2 = self.scaler2.transform(x_filtered2)
            ypred2 = self.model2.predict(x_scaled2) # 0: Circle, 1: Diagonal
            
            # Circle
            circle_mask = (ypred2 == 0)
            circle_indices = complex_indices[circle_mask]
            ypred[circle_indices] = self.cid
            
            # Diagonal
            diag_mask = (ypred2 == 1)
            diag_indices = complex_indices[diag_mask]
            
            if len(diag_indices) > 0:
                xtestdiag = [xorig[i] for i in diag_indices]
                xtestdiag = pad_sequences(xtestdiag, padding='post', dtype='float32', value=np.nan)
                
                # DTW
                ypred3 = self.model3.predict(xtestdiag)
                ypred[diag_indices] = ypred3

        return ypred

    def getrules(self, featurenames):
        rules1 = "Linear SVM (H vs V vs Complex)"
        rules2 = "Linear SVM (Circle vs Diagonals)"
        rules3 = f"KNN-DTW (n={self.model3.n_neighbors})"
        return rules1, rules2, rules3

def visualize_model1(model, features, y_true):
    x_filtered = model._filter_features(features, model.drop_indices_model1)
    x_scaled = model.scaler1.transform(x_filtered)

    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x_scaled)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1
    y_min, y_max = x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1
    z_min, z_max = x_pca[:, 2].min() - 1, x_pca[:, 2].max() + 1

    w_all = model.model1.coef_
    b_all = model.model1.intercept_

    plane_colors = ['cyan', 'purple', 'orange']
    plane_labels = ['Horiz vs Vert', 'Horiz vs Complex', 'Vert vs Complex']

    for i in range(min(3, len(w_all))):
        w_pca = w_all[i] @ pca.components_.T
        b_pca = b_all[i] + np.dot(w_all[i], pca.mean_)

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                             np.linspace(y_min, y_max, 10))

        if np.abs(w_pca[2]) > np.abs(w_pca[0]) and np.abs(w_pca[2]) > np.abs(w_pca[1]):
            z_plane = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
            mask = (z_plane >= z_min) & (z_plane <= z_max)
            if np.any(mask):
                ax.plot_surface(xx, yy, z_plane, alpha=0.2, color=plane_colors[i])
                ax.plot([], [], [], color=plane_colors[i], alpha=0.5, label=f'Boundary: {plane_labels[i]}')
        
        elif np.abs(w_pca[1]) > np.abs(w_pca[0]):
            xx, zz = np.meshgrid(np.linspace(x_min, x_max, 10),
                                 np.linspace(z_min, z_max, 10))
            y_plane = -(w_pca[0] * xx + w_pca[2] * zz + b_pca) / w_pca[1]
            mask = (y_plane >= y_min) & (y_plane <= y_max)
            if np.any(mask):
                ax.plot_surface(xx, y_plane, zz, alpha=0.2, color=plane_colors[i])
                ax.plot([], [], [], color=plane_colors[i], alpha=0.5, label=f'Boundary: {plane_labels[i]}')
        
        else:
            yy, zz = np.meshgrid(np.linspace(y_min, y_max, 10),
                                 np.linspace(z_min, z_max, 10))
            x_plane = -(w_pca[1] * yy + w_pca[2] * zz + b_pca) / w_pca[0]
            mask = (x_plane >= x_min) & (x_plane <= x_max)
            if np.any(mask):
                ax.plot_surface(x_plane, yy, zz, alpha=0.2, color=plane_colors[i])
                ax.plot([], [], [], color=plane_colors[i], alpha=0.5, label=f'Boundary: {plane_labels[i]}')

    colordict = {
        label['horizontal']: 'blue',
        label['vertical']: 'cyan',
        label['circle']: 'red',
        label['diagonal_left']: 'orange',
        label['diagonal_right']: 'gold'
    }

    ypredbinary = model.model1.predict(x_scaled)

    for classname, classid in label.items():
        idxs = np.where(y_true == classid)[0]
        if len(idxs) == 0: continue

        target = 0 if classid == label['horizontal'] else (1 if classid == label['vertical'] else 2)

        correct_idxs = idxs[ypredbinary[idxs] == target]
        wrong_idxs = idxs[ypredbinary[idxs] != target]

        if len(correct_idxs) > 0:
            ax.scatter(x_pca[correct_idxs, 0], x_pca[correct_idxs, 1], x_pca[correct_idxs, 2],
                       c=colordict[classid], marker='o', s=60, edgecolor='black', alpha=0.9,
                       label=f'{classname}')

        if len(wrong_idxs) > 0:
            ax.scatter(x_pca[wrong_idxs, 0], x_pca[wrong_idxs, 1], x_pca[wrong_idxs, 2],
                       c=colordict[classid], marker='X', s=100, edgecolor='black', linewidth=2,
                       label=f'{classname} (Wrong in M1)')

    ax.set_title('SVM H vs V vs Complex', fontsize=15)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    plt.show()

def visualize_model2(model, features, y_true):
    complex_labels = [label['circle'], label['diagonal_left'], label['diagonal_right']]
    mask = np.isin(y_true, complex_labels)
    
    x_subset = features[mask]
    y_subset = y_true[mask]
    
    if len(x_subset) == 0: return

    y_binary = np.where(y_subset == label['circle'], 0, 1)

    x_filtered = model._filter_features(x_subset, model.drop_indices_model2)
    x_scaled = model.scaler2.transform(x_filtered)

    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x_scaled)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1
    y_min, y_max = x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1
    z_min, z_max = x_pca[:, 2].min() - 1, x_pca[:, 2].max() + 1

    w = model.model2.coef_[0]
    b = model.model2.intercept_[0]
    
    w_pca = w @ pca.components_.T
    b_pca = b + np.dot(w, pca.mean_)

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

    if np.abs(w_pca[2]) > np.abs(w_pca[0]) and np.abs(w_pca[2]) > np.abs(w_pca[1]):
        z_plane = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
        mask_plane = (z_plane >= z_min) & (z_plane <= z_max)
        if np.any(mask_plane):
            ax.plot_surface(xx, yy, z_plane, alpha=0.3, color='green')

    colordict = {0: 'red', 1: 'orange'}
    labeldict = {0: 'Circle', 1: 'Diagonal'}
    
    ypred_subset = model.model2.predict(x_scaled)

    for i in [0, 1]: 
        idxs = np.where(y_binary == i)[0]
        if len(idxs) == 0: continue

        correct = idxs[ypred_subset[idxs] == i]
        wrong = idxs[ypred_subset[idxs] != i]

        if len(correct) > 0:
            ax.scatter(x_pca[correct, 0], x_pca[correct, 1], x_pca[correct, 2],
                       c=colordict[i], marker='o', s=60, edgecolor='black', alpha=0.9,
                       label=labeldict[i])
        if len(wrong) > 0:
            ax.scatter(x_pca[wrong, 0], x_pca[wrong, 1], x_pca[wrong, 2],
                       c=colordict[i], marker='X', s=100, edgecolor='black', linewidth=2,
                       label=f'{labeldict[i]} (Wrong)')

    ax.set_title('SVM Circle vs Diagonal', fontsize=15)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.legend()
    plt.tight_layout()
    plt.show()

data = os.path.join(project_root, 'data')
naugment = 9
testsize = 0.2
randomstate = 42
classnames = list(label.keys())

xorig, yorig = load(data)

if len(xorig) == 0:
    print("로드안됨")
    exit()

xtrainorig, xtestorig, ytrainorig, ytest = train_test_split(
    xorig, yorig,
    test_size=testsize,
    random_state=randomstate,
    stratify=yorig)

xauglist, ytrainaug = augmentdata(
    xtrainorig,
    ytrainorig,
    n=naugment)

xtrain, featurenames = extractfeatures(xauglist)
xtest, _ = extractfeatures(xtestorig)

print(f"훈련 데이터 x {xtrain.shape}, y {ytrainaug.shape}")
print(f"테스트 데이터 x {xtest.shape}, y {ytest.shape}")

model = HybridClassifier(maxdepthsimple=3, maxdepthcomplex=3, maxneighbor2=3, randomstate=42)

model.fit(xtrain, ytrainaug, xauglist)

save = 'mainmodel.joblib'
joblib.dump(model, save)
print(f"\n모델 저장 완료: {save}")

print("\n모델 평가")
ypred = model.predict(xtest, xtestorig)
accuracy = np.mean(ypred == ytest)

print(f"\n테스트 정확도: {accuracy:.4f}")
print(classification_report(ytest, ypred, target_names=classnames, zero_division=0))

r1, r2, r3 = model.getrules(featurenames)
print(r1)
print(r2)
print(r3)
visualize_model1(model, xtrain, ytrainaug)
visualize_model2(model, xtrain, ytrainaug)