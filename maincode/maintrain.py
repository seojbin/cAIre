import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesResampler
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
        self.scaler = StandardScaler()

        self.model1 = SVC(kernel='linear', random_state=self.randomstate)

        self.model3 = DecisionTreeClassifier(max_depth=maxdepthcomplex, random_state=self.randomstate)
        self.model4 = KNeighborsTimeSeriesClassifier(n_neighbors=maxneighbor2, metric='dtw')

        self.cid = label['circle']
        self.cdl = label['diagonal_left']
        self.cdr = label['diagonal_right']
        self.cho = label['horizontal']
        self.cve = label['vertical']

        self.complexlabels = [self.cid, self.cdl, self.cdr]
        self.simplelabels = [self.cho, self.cve]
        self.diagonallabels = [self.cdl, self.cdr]
        self.drop_indices_model1 = [3,4,5,9,10,12]

    def _filter_features(self, x):
        return np.delete(x, self.drop_indices_model1, axis=1)

    def fit(self, x, y, xorig):
        x_filtered = self._filter_features(x)
        self.scaler.fit(x_filtered)
        x_scaled = self.scaler.transform(x_filtered)

        ytrain1 = np.full(y.shape, 2)
        ytrain1[y == self.cho] = 0
        ytrain1[y == self.cve] = 1

        complexmask = (ytrain1 == 2)
        xtrain3 = x[complexmask]
        ytrain3_orig = y[complexmask]
        ytrain3 = np.where(np.isin(ytrain3_orig, self.diagonallabels), 1, 0)

        diagonalmask = np.isin(y, self.diagonallabels)
        xtrain4 = [xorig[i] for i in range(len(xorig)) if diagonalmask[i]]
        xtrain4 = pad_sequences(xtrain4, padding='post', dtype='float32', value=np.nan)
        ytrain4 = y[diagonalmask]

        self.model1.fit(x_scaled, ytrain1)
        self.model3.fit(xtrain3, ytrain3)
        self.model4.fit(xtrain4, ytrain4)

    def predict(self, x, xorig):
        x_filtered = self._filter_features(x)
        x_scaled = self.scaler.transform(x_filtered)

        ypred1 = self.model1.predict(x_scaled)
        ypred = np.zeros(len(x), dtype=int)

        ypred[ypred1 == 0] = self.cho
        ypred[ypred1 == 1] = self.cve

        testcomplexmask = (ypred1 == 2)

        xtestcomplex = x[testcomplexmask]
        if xtestcomplex.shape[0] > 0:
            ypred3 = self.model3.predict(xtestcomplex)

            complex_indices = np.where(testcomplexmask)[0]
            mask3circle = (ypred3 == 0)
            mask3diag = (ypred3 == 1)

            circle_indices_to_update = complex_indices[mask3circle]
            if len(circle_indices_to_update) > 0:
                ypred[circle_indices_to_update] = self.cid

            diag_indices_in_subset = complex_indices[mask3diag]
            xtestdiag = [xorig[i] for i in diag_indices_in_subset]

            if len(xtestdiag) > 0:
                xtestdiag = pad_sequences(xtestdiag, padding='post', dtype='float32', value=np.nan)
                ypred4 = self.model4.predict(xtestdiag)

                if len(diag_indices_in_subset) == len(ypred4):
                    ypred[diag_indices_in_subset] = ypred4
                else:
                    print("predict 길이 불일치")

        return ypred

    def getrules(self, featurenames):
        rules1 = "Linear SVM "
        rules3 = export_text(self.model3, feature_names=featurenames, class_names=['circle', 'diagonal'])
        rules4 = f"KNeighborsTimeSeriesClassifier(n_neighbors={self.model4.n_neighbors}, metric='{self.model4.metric}')"
        return rules1, rules3, rules4


def visualize_model1(model, features, y_true):
    x_filtered = model._filter_features(features)
    x_scaled = model.scaler.transform(x_filtered)

    #3차원 PCA 변환
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x_scaled)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 경계면
    # SVM 가중치(w) 절편(b) PCA 공간 투영

    # 축 범위 설정 (메쉬그리드 생성용)
    x_min, x_max = x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1
    y_min, y_max = x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1
    z_min, z_max = x_pca[:, 2].min() - 1, x_pca[:, 2].max() + 1

    # SVM 모델 파라미터 가져오기
    w_all = model.model1.coef_
    b_all = model.model1.intercept_

    plane_colors = ['cyan', 'purple', 'orange']
    plane_labels = ['Horiz vs Vert', 'Horiz vs Complex', 'Vert vs Complex']

    for i in range(3):
        w_pca = w_all[i] @ pca.components_.T
        b_pca = b_all[i] + np.dot(w_all[i], pca.mean_)

        # 평면 그리기
        # w0*x + w1*y + w2*z + b=0

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                             np.linspace(y_min, y_max, 10))

        # z
        if np.abs(w_pca[2]) > np.abs(w_pca[0]) and np.abs(w_pca[2]) > np.abs(w_pca[1]):
            z_plane = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
            # z 범위 안에 들어오는 것만 그리기 (시각화 깔끔하게)
            mask = (z_plane >= z_min) & (z_plane <= z_max)
            if np.any(mask):
                ax.plot_surface(xx, yy, z_plane, alpha=0.2, color=plane_colors[i])
                # 범례용 더미 플롯
                ax.plot([], [], [], color=plane_colors[i], alpha=0.5, label=f'Boundary: {plane_labels[i]}')

        # y
        elif np.abs(w_pca[1]) > np.abs(w_pca[0]):
            xx, zz = np.meshgrid(np.linspace(x_min, x_max, 10),
                                 np.linspace(z_min, z_max, 10))
            y_plane = -(w_pca[0] * xx + w_pca[2] * zz + b_pca) / w_pca[1]
            mask = (y_plane >= y_min) & (y_plane <= y_max)
            if np.any(mask):
                ax.plot_surface(xx, y_plane, zz, alpha=0.2, color=plane_colors[i])
                ax.plot([], [], [], color=plane_colors[i], alpha=0.5, label=f'Boundary: {plane_labels[i]}')

        # x
        else:
            yy, zz = np.meshgrid(np.linspace(y_min, y_max, 10),
                                 np.linspace(z_min, z_max, 10))
            x_plane = -(w_pca[1] * yy + w_pca[2] * zz + b_pca) / w_pca[0]
            mask = (x_plane >= x_min) & (x_plane <= x_max)
            if np.any(mask):
                ax.plot_surface(x_plane, yy, zz, alpha=0.2, color=plane_colors[i])
                ax.plot([], [], [], color=plane_colors[i], alpha=0.5, label=f'Boundary: {plane_labels[i]}')

    # 산점도
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

        # SVM 매핑
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
                       label=f'{classname} (Wrong)')

    ax.set_title('SVM 3D PCA', fontsize=15)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.view_init(elev=25, azim=135)  # 초기 각도
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

rules1, rules3, rules4 = model.getrules(featurenames)

print("Model 1 (SVM)")
print(rules1)

print("circle vs diagonal")
print(rules3)

print("diagonal_left vs diagonal_right")
print(rules4)
visualize_model1(model, xtrain, ytrainaug)