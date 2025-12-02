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
    print("전처리 파일(preprocess.py, feature_extractor.py)이 없습니다.")
    exit()


class HybridClassifier:
    def __init__(self, maxdepthsimple=3, maxdepthcomplex=3, maxneighbor2=3, randomstate=42):
        self.randomstate = randomstate
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()

        # Model 1, 2, 3: SVM (계층형)
        self.model1 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced')
        self.model2 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced')
        self.model3 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced')

        # Model 4: KNN-DTW (Left vs Right)
        # 증강 데이터를 학습하므로 K=3이 적절합니다.
        self.model4 = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric='dtw')

        self.cid = label['circle']
        self.cdl = label['diagonal_left']
        self.cdr = label['diagonal_right']
        self.cho = label['horizontal']
        self.cve = label['vertical']

        # [Additive Feature Selection] 사용할 피쳐 인덱스
        self.select_indices_model1 = [1, 2, 6, 7, 8, 11]
        self.select_indices_model2 = [1, 2, 6, 7, 8, 11]
        self.select_indices_model3 = [1, 7, 9, 11, 12]

    def _filter_features(self, x, indices):
        return x[:, indices]

    def fit(self, x, y, xorig):
        """
        x: 증강된 피쳐 (SVM용)
        y: 증강된 라벨
        xorig: 증강된 시계열 (KNN용)
        """

        # --- Model 1: H vs Rest ---
        x_f1 = self._filter_features(x, self.select_indices_model1)
        self.scaler1.fit(x_f1)
        x_s1 = self.scaler1.transform(x_f1)

        y_m1 = np.where(y == self.cho, 0, 1)
        self.model1.fit(x_s1, y_m1)

        # --- Model 2: V vs Complex ---
        mask_m2 = (y != self.cho)
        x_m2 = x[mask_m2]
        y_m2 = y[mask_m2]

        if len(x_m2) > 0:
            x_f2 = self._filter_features(x_m2, self.select_indices_model2)
            self.scaler2.fit(x_f2)
            x_s2 = self.scaler2.transform(x_f2)

            y_train2 = np.where(y_m2 == self.cve, 0, 1)
            self.model2.fit(x_s2, y_train2)

            # --- Model 3: Circle vs Diagonal ---
            mask_m3 = (y_m2 != self.cve)
            x_m3 = x_m2[mask_m3]
            y_m3 = y_m2[mask_m3]

            if len(x_m3) > 0:
                x_f3 = self._filter_features(x_m3, self.select_indices_model3)
                self.scaler3.fit(x_f3)
                x_s3 = self.scaler3.transform(x_f3)

                y_train3 = np.where(y_m3 == self.cid, 0, 1)
                self.model3.fit(x_s3, y_train3)

                # --- Model 4: KNN (Left vs Right) ---
                # 증강된 시계열 데이터(xorig)에서 대각선만 추출하여 학습
                mask_m4 = (y_m3 != self.cid)

                # 인덱싱을 위해 전체 데이터에서 다시 필터링
                diag_targets = [self.cdl, self.cdr]
                diag_mask_total = np.isin(y, diag_targets)

                xtrain4 = [xorig[i] for i in range(len(xorig)) if diag_mask_total[i]]
                ytrain4 = y[diag_mask_total]

                if len(xtrain4) > 0:
                    xtrain4_padded = pad_sequences(xtrain4, padding='post', dtype='float32', value=np.nan)
                    self.model4.fit(xtrain4_padded, ytrain4)
                    print(f"Model 4 (KNN) Fitted with {len(xtrain4)} augmented samples.")

    def predict(self, x, xorig):
        ypred = np.zeros(len(x), dtype=int)

        # M1
        x_f1 = self._filter_features(x, self.select_indices_model1)
        x_s1 = self.scaler1.transform(x_f1)
        p1 = self.model1.predict(x_s1)

        ypred[p1 == 0] = self.cho
        rest_idx = np.where(p1 == 1)[0]
        if len(rest_idx) == 0: return ypred

        # M2
        x_rest = x[rest_idx]
        x_f2 = self._filter_features(x_rest, self.select_indices_model2)
        x_s2 = self.scaler2.transform(x_f2)
        p2 = self.model2.predict(x_s2)

        v_idx = rest_idx[p2 == 0]
        ypred[v_idx] = self.cve
        comp_idx = rest_idx[p2 == 1]
        if len(comp_idx) == 0: return ypred

        # M3
        x_comp = x[comp_idx]
        x_f3 = self._filter_features(x_comp, self.select_indices_model3)
        x_s3 = self.scaler3.transform(x_f3)
        p3 = self.model3.predict(x_s3)

        c_idx = comp_idx[p3 == 0]
        ypred[c_idx] = self.cid
        diag_idx = comp_idx[p3 == 1]
        if len(diag_idx) == 0: return ypred

        # M4 (KNN)
        x_diag_ts = [xorig[i] for i in diag_idx]
        train_maxlen = self.model4._X_fit.shape[1]
        x_diag_ts = pad_sequences(x_diag_ts, maxlen=train_maxlen, padding='post', dtype='float32', value=np.nan)

        p4 = self.model4.predict(x_diag_ts)
        ypred[diag_idx] = p4

        return ypred

    def getrules(self, featurenames):
        rules1 = "Model 1: Linear SVM (H vs V vs Complex)"
        rules2 = "Model 2: Linear SVM (V vs Complex)"
        rules3 = "Model 3: Linear SVM (Circle vs Diagonal)"
        rules4 = f"Model 4: KNN-DTW (n={self.model4.n_neighbors})"
        return rules1, rules2, rules3, rules4


def visualize_model_svm(model_clf, scaler, select_indices, x_data, y_true, title, label_map, feature_names):
    if len(x_data) == 0: return
    x_f = x_data[:, select_indices]
    x_s = scaler.transform(x_f)
    pca = PCA(n_components=3).fit(x_s)
    x_pca = pca.transform(x_s)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    w = model_clf.coef_[0]
    b = model_clf.intercept_[0]
    used_names = [feature_names[i] for i in select_indices]
    eq = " + ".join([f"({w[i]:.2f}*{used_names[i]})" for i in range(len(w))])
    print(f"\n[{title}] Hyperplane: {eq} + {b:.2f} = 0")

    w_pca = w @ pca.components_.T
    b_pca = b + np.dot(w, pca.mean_)
    x_min, x_max = x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1
    y_min, y_max = x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

    if abs(w_pca[2]) > 0.001:
        z = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
        ax.plot_surface(xx, yy, z, alpha=0.2, color='gray')

    preds = model_clf.predict(x_s)
    colors = ['blue', 'red']
    for cls, target in label_map.items():
        idxs = np.where(y_true == cls)[0]
        correct = idxs[preds[idxs] == target]
        wrong = idxs[preds[idxs] != target]
        if len(correct) > 0: ax.scatter(x_pca[correct, 0], x_pca[correct, 1], x_pca[correct, 2], c=colors[target],
                                        marker='o', s=40, alpha=0.6, label=f'Class {cls}')
        if len(wrong) > 0: ax.scatter(x_pca[wrong, 0], x_pca[wrong, 1], x_pca[wrong, 2], c=colors[target], marker='x',
                                      s=100, label=f'Wrong {cls}')
    ax.set_title(title);
    plt.legend();
    plt.show()


# KNN은 초평면이 없으므로 데이터 분포만 시각화 (PCA)
def visualize_knn_pca(model, x_data, y_true, title, label_map):
    # Model 4 데이터 필터링 (대각선만)
    diag_indices = [i for i, lbl in enumerate(y_true) if lbl in label_map]
    if len(diag_indices) == 0: return

    # KNN 학습 데이터 (증강된 것들)
    X_train_raw = model.model4._X_fit
    y_train = model.model4._y

    # PCA를 위해 시계열을 1차원으로 펼침 (Flatten) or 평균 특징 사용
    # 여기서는 간단히 평균/std 등 통계값 대신 feature extractor를 재사용할 수 없으니(X_train_raw가 이미 전처리됨)
    # 시각화 용도로만 X_train의 평균 위치값 사용
    X_train_mean = np.nanmean(X_train_raw, axis=1)  # (N, 3)

    pca = PCA(n_components=2)  # 2D로 표현
    X_pca = pca.fit_transform(X_train_mean)

    plt.figure(figsize=(8, 6))
    colors = {label['diagonal_left']: 'orange', label['diagonal_right']: 'gold'}

    for lbl in [label['diagonal_left'], label['diagonal_right']]:
        idxs = np.where(y_train == lbl)[0]
        plt.scatter(X_pca[idxs, 0], X_pca[idxs, 1], c=colors[lbl], label=f'Train {lbl}', alpha=0.5)

    plt.title(f"{title} (Data Distribution via PCA)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = os.path.join(project_root, 'data')
    naugment = 9
    xorig, yorig = load(data)

    xtrainorig, xtestorig, ytrainorig, ytest = train_test_split(xorig, yorig, test_size=0.2, stratify=yorig,
                                                                random_state=42)
    xauglist, ytrainaug = augmentdata(xtrainorig, ytrainorig, n=naugment)
    xtrain, featurenames = extractfeatures(xauglist)
    xtest, _ = extractfeatures(xtestorig)

    model = HybridClassifier()
    # 증강 데이터로 학습
    model.fit(xtrain, ytrainaug, xauglist)

    joblib.dump(model, 'mainmodel3.joblib')

    ypred = model.predict(xtest, xtestorig)
    print(classification_report(ytest, ypred, target_names=list(label.keys()), zero_division=0))

    # --- 시각화 호출 (누락되었던 부분 복구) ---
    print("\nVisualizing Models...")

    # Model 1
    visualize_model_svm(model.model1, model.scaler1, model.select_indices_model1, xtrain, ytrainaug,
                        "M1: Horiz vs Rest",
                        {label['horizontal']: 0, label['vertical']: 1, label['circle']: 1, label['diagonal_left']: 1,
                         label['diagonal_right']: 1}, featurenames)

    # Model 2
    m2_mask = (ytrainaug != label['horizontal'])
    visualize_model_svm(model.model2, model.scaler2, model.select_indices_model2, xtrain[m2_mask], ytrainaug[m2_mask],
                        "M2: Vert vs Complex", {label['vertical']: 0, label['circle']: 1, label['diagonal_left']: 1,
                                                label['diagonal_right']: 1}, featurenames)

    # Model 3
    m3_mask = m2_mask & (ytrainaug != label['vertical'])
    visualize_model_svm(model.model3, model.scaler3, model.select_indices_model3, xtrain[m3_mask], ytrainaug[m3_mask],
                        "M3: Circle vs Diag",
                        {label['circle']: 0, label['diagonal_left']: 1, label['diagonal_right']: 1}, featurenames)

    # Model 4 (KNN) - 분포 확인용
    visualize_knn_pca(model, xauglist, ytrainaug, "M4: Left vs Right (Augmented Data)",
                      {label['diagonal_left']: 0, label['diagonal_right']: 1})