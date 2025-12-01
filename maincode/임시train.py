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
        self.scaler3 = StandardScaler()

        # Model 1: H vs V vs Complex (Linear SVM)
        self.model1 = SVC(kernel='linear', random_state=self.randomstate)

        # Model 2: Circle vs Diagonal(통합) (Linear SVM)
        self.model2 = SVC(kernel='linear', random_state=self.randomstate)

        # Model 3: Diagonal Left vs Diagonal Right (Linear SVM)
        self.model3 = SVC(kernel='linear', random_state=self.randomstate)

        self.cid = label['circle']
        self.cdl = label['diagonal_left']
        self.cdr = label['diagonal_right']
        self.cho = label['horizontal']
        self.cve = label['vertical']

        self.diagonallabels = [self.cdl, self.cdr]
        self.complexlabels = [self.cid, self.cdl, self.cdr]

        # 피쳐 인덱스 (feature_extractor.py 기준)
        # 0:X_std, 1:Y_std, 2:Z_std, 3:length, 4:disp, 5:len_disp,
        # 6:rX, 7:rY, 8:rZ, 9:ratio, 10:jerk, 11:area, 12:slope, 13:corr

        self.drop_indices_model1 = [3, 4, 5, 9, 10, 12, 13]
        self.drop_indices_model2 = [3, 4, 5, 6, 8, 10, 13]
        self.drop_indices_model3 = [0, 3, 4, 5, 6, 9, 10]

    def _filter_features(self, x, indices):
        return np.delete(x, indices, axis=1)

    def fit(self, x, y, xorig):
        # --- Model 1 ---
        x_filtered1 = self._filter_features(x, self.drop_indices_model1)
        self.scaler1.fit(x_filtered1)
        x_scaled1 = self.scaler1.transform(x_filtered1)

        ytrain1 = np.full(y.shape, 2)
        ytrain1[y == self.cho] = 0
        ytrain1[y == self.cve] = 1

        self.model1.fit(x_scaled1, ytrain1)

        # --- Model 2 (Circle vs Diagonal) ---
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

        # --- Model 3 (Diag Left vs Diag Right) ---
        diag_mask = (ytrain2 == 1)
        xtrain3 = xtrain2[diag_mask]
        ytrain3_orig = ytrain2_orig[diag_mask]

        if len(xtrain3) > 0:
            x_filtered3 = self._filter_features(xtrain3, self.drop_indices_model3)
            self.scaler3.fit(x_filtered3)
            x_scaled3 = self.scaler3.transform(x_filtered3)
            self.model3.fit(x_scaled3, ytrain3_orig)

    def predict(self, x, xorig):
        # Model 1
        x_filtered1 = self._filter_features(x, self.drop_indices_model1)
        x_scaled1 = self.scaler1.transform(x_filtered1)
        ypred1 = self.model1.predict(x_scaled1)

        ypred = np.zeros(len(x), dtype=int)
        ypred[ypred1 == 0] = self.cho
        ypred[ypred1 == 1] = self.cve

        # Model 2
        complexmask = (ypred1 == 2)
        xtestcomplex = x[complexmask]
        complex_indices = np.where(complexmask)[0]

        if len(complex_indices) > 0:
            x_filtered2 = self._filter_features(xtestcomplex, self.drop_indices_model2)
            x_scaled2 = self.scaler2.transform(x_filtered2)
            ypred2 = self.model2.predict(x_scaled2)  # 0: Circle, 1: Diagonal

            # Circle 할당
            circle_mask = (ypred2 == 0)
            circle_indices = complex_indices[circle_mask]
            ypred[circle_indices] = self.cid

            # Model 3 (Diagonal)
            diag_mask = (ypred2 == 1)
            diag_indices = complex_indices[diag_mask]
            xtestdiag = xtestcomplex[diag_mask]

            if len(xtestdiag) > 0:
                x_filtered3 = self._filter_features(xtestdiag, self.drop_indices_model3)
                x_scaled3 = self.scaler3.transform(x_filtered3)
                ypred3 = self.model3.predict(x_scaled3)
                ypred[diag_indices] = ypred3

        return ypred

    def getrules(self, featurenames):
        rules1 = "Linear SVM (H vs V vs Complex)"
        rules2 = "Linear SVM (Circle vs Diagonals)"
        rules3 = "Linear SVM (Diag Left vs Diag Right)"
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

    for i in range(min(3, len(w_all))):
        w_pca = w_all[i] @ pca.components_.T
        b_pca = b_all[i] + np.dot(w_all[i], pca.mean_)

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

        if np.abs(w_pca[2]) > np.abs(w_pca[0]) and np.abs(w_pca[2]) > np.abs(w_pca[1]):
            z_plane = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
            mask = (z_plane >= z_min) & (z_plane <= z_max)
            if np.any(mask):
                ax.plot_surface(xx, yy, z_plane, alpha=0.2, color=plane_colors[i])
        elif np.abs(w_pca[1]) > np.abs(w_pca[0]):
            xx, zz = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
            y_plane = -(w_pca[0] * xx + w_pca[2] * zz + b_pca) / w_pca[1]
            mask = (y_plane >= y_min) & (y_plane <= y_max)
            if np.any(mask):
                ax.plot_surface(xx, y_plane, zz, alpha=0.2, color=plane_colors[i])
        else:
            yy, zz = np.meshgrid(np.linspace(y_min, y_max, 10), np.linspace(z_min, z_max, 10))
            x_plane = -(w_pca[1] * yy + w_pca[2] * zz + b_pca) / w_pca[0]
            mask = (x_plane >= x_min) & (x_plane <= x_max)
            if np.any(mask):
                ax.plot_surface(x_plane, yy, zz, alpha=0.2, color=plane_colors[i])

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
                       c=colordict[classid], marker='o', s=60, edgecolor='black', alpha=0.9, label=f'{classname}')
        if len(wrong_idxs) > 0:
            ax.scatter(x_pca[wrong_idxs, 0], x_pca[wrong_idxs, 1], x_pca[wrong_idxs, 2],
                       c=colordict[classid], marker='X', s=100, edgecolor='black', linewidth=2,
                       label=f'{classname} (Wrong)')

    ax.set_title('SVM Model 1: H vs V vs Complex', fontsize=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def visualize_model2(model, features, y_true):
    complex_labels = [label['circle'], label['diagonal_left'], label['diagonal_right']]
    mask = np.isin(y_true, complex_labels)

    x_subset = features[mask]
    y_subset = y_true[mask]
    if len(x_subset) == 0: return

    x_filtered = model._filter_features(x_subset, model.drop_indices_model2)
    x_scaled = model.scaler2.transform(x_filtered)

    # 0: Circle, 1: Diagonal
    y_binary = np.where(y_subset == label['circle'], 0, 1)

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
    elif np.abs(w_pca[1]) > np.abs(w_pca[0]):
        xx, zz = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
        y_plane = -(w_pca[0] * xx + w_pca[2] * zz + b_pca) / w_pca[1]
        mask_plane = (y_plane >= y_min) & (y_plane <= y_max)
        if np.any(mask_plane):
            ax.plot_surface(xx, y_plane, zz, alpha=0.3, color='green')
    else:
        yy, zz = np.meshgrid(np.linspace(y_min, y_max, 10), np.linspace(z_min, z_max, 10))
        x_plane = -(w_pca[1] * yy + w_pca[2] * zz + b_pca) / w_pca[0]
        mask_plane = (x_plane >= x_min) & (x_plane <= x_max)
        if np.any(mask_plane):
            ax.plot_surface(x_plane, yy, zz, alpha=0.3, color='green')

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
                       c=colordict[i], marker='o', s=60, edgecolor='black', alpha=0.9, label=labeldict[i])
        if len(wrong) > 0:
            ax.scatter(x_pca[wrong, 0], x_pca[wrong, 1], x_pca[wrong, 2],
                       c=colordict[i], marker='X', s=100, edgecolor='black', linewidth=2,
                       label=f'{labeldict[i]} (Wrong)')

    ax.set_title('SVM Model 2: Circle vs Diagonal', fontsize=15)
    ax.legend()
    plt.show()


def visualize_model3(model, features, y_true):
    diag_labels = [label['diagonal_left'], label['diagonal_right']]
    mask = np.isin(y_true, diag_labels)

    x_subset = features[mask]
    y_subset = y_true[mask]
    if len(x_subset) == 0: return

    x_filtered = model._filter_features(x_subset, model.drop_indices_model3)
    x_scaled = model.scaler3.transform(x_filtered)

    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x_scaled)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1
    y_min, y_max = x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1
    z_min, z_max = x_pca[:, 2].min() - 1, x_pca[:, 2].max() + 1

    w = model.model3.coef_[0]
    b = model.model3.intercept_[0]
    w_pca = w @ pca.components_.T
    b_pca = b + np.dot(w, pca.mean_)

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

    if np.abs(w_pca[2]) > np.abs(w_pca[0]) and np.abs(w_pca[2]) > np.abs(w_pca[1]):
        z_plane = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
        mask_plane = (z_plane >= z_min) & (z_plane <= z_max)
        if np.any(mask_plane):
            ax.plot_surface(xx, yy, z_plane, alpha=0.3, color='gold')
    elif np.abs(w_pca[1]) > np.abs(w_pca[0]):
        xx, zz = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
        y_plane = -(w_pca[0] * xx + w_pca[2] * zz + b_pca) / w_pca[1]
        mask_plane = (y_plane >= y_min) & (y_plane <= y_max)
        if np.any(mask_plane):
            ax.plot_surface(xx, y_plane, zz, alpha=0.3, color='gold')
    else:
        yy, zz = np.meshgrid(np.linspace(y_min, y_max, 10), np.linspace(z_min, z_max, 10))
        x_plane = -(w_pca[1] * yy + w_pca[2] * zz + b_pca) / w_pca[0]
        mask_plane = (x_plane >= x_min) & (x_plane <= x_max)
        if np.any(mask_plane):
            ax.plot_surface(x_plane, yy, zz, alpha=0.3, color='gold')

    colordict = {label['diagonal_left']: 'orange', label['diagonal_right']: 'gold'}

    ypred_subset = model.model3.predict(x_scaled)

    for classname, classid in label.items():
        if classid not in diag_labels: continue
        idxs = np.where(y_subset == classid)[0]
        if len(idxs) == 0: continue

        correct = idxs[ypred_subset[idxs] == classid]
        wrong = idxs[ypred_subset[idxs] != classid]

        if len(correct) > 0:
            ax.scatter(x_pca[correct, 0], x_pca[correct, 1], x_pca[correct, 2],
                       c=colordict[classid], marker='o', s=60, edgecolor='black', alpha=0.9, label=f'{classname}')
        if len(wrong) > 0:
            ax.scatter(x_pca[wrong, 0], x_pca[wrong, 1], x_pca[wrong, 2],
                       c=colordict[classid], marker='X', s=100, edgecolor='black', linewidth=2,
                       label=f'{classname} (Wrong)')

    ax.set_title('SVM Model 3: Diag Left vs Right', fontsize=15)
    ax.legend()
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

save = 'mainmodel2.joblib'
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
visualize_model3(model, xtrain, ytrainaug)