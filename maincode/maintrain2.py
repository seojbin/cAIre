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

        # 피쳐 인덱스
        self.select_indices_model1 = [1, 2, 6, 7, 8, 11, 14]
        self.select_indices_model2 = [1, 2, 6, 7, 8, 11, 14]
        self.select_indices_model3 = [1, 7, 9, 11, 12, 0]
        self.select_indices_model4 = [14, 15, 6]

    def _filter_features(self, x, indices):
        return x[:, indices]

    def fit(self, x, y, xorig):
        # M1
        x_f1 = self._filter_features(x, self.select_indices_model1)
        self.scaler1.fit(x_f1)
        x_s1 = self.scaler1.transform(x_f1)
        y_m1 = np.where(y == self.cho, 0, 1)
        self.model1.fit(x_s1, y_m1)

        # M2
        mask_m2 = (y != self.cho)
        x_m2 = x[mask_m2]
        y_m2 = y[mask_m2]
        if len(x_m2) > 0:
            x_f2 = self._filter_features(x_m2, self.select_indices_model2)
            self.scaler2.fit(x_f2)
            x_s2 = self.scaler2.transform(x_f2)
            y_train2 = np.where(y_m2 == self.cve, 0, 1)
            self.model2.fit(x_s2, y_train2)

            # M3
            mask_m3 = (y_m2 != self.cve)
            x_m3 = x_m2[mask_m3]
            y_m3 = y_m2[mask_m3]
            if len(x_m3) > 0:
                x_f3 = self._filter_features(x_m3, self.select_indices_model3)
                self.scaler3.fit(x_f3)
                x_s3 = self.scaler3.transform(x_f3)
                y_train3 = np.where(y_m3 == self.cid, 0, 1)
                self.model3.fit(x_s3, y_train3)

                # M4
                mask_m4 = (y_m3 != self.cid)
                x_m4 = x_m3[mask_m4]
                y_m4 = y_m3[mask_m4]
                if len(x_m4) > 0:
                    x_f4 = self._filter_features(x_m4, self.select_indices_model4)
                    self.scaler4.fit(x_f4)
                    x_s4 = self.scaler4.transform(x_f4)
                    y_train4 = np.where(y_m4 == self.cdl, 0, 1)
                    self.model4.fit(x_s4, y_train4)

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
        ypred[rest_idx[p2 == 0]] = self.cve

        comp_idx = rest_idx[p2 == 1]
        if len(comp_idx) == 0: return ypred

        # M3
        x_comp = x[comp_idx]
        x_f3 = self._filter_features(x_comp, self.select_indices_model3)
        x_s3 = self.scaler3.transform(x_f3)
        p3 = self.model3.predict(x_s3)
        ypred[comp_idx[p3 == 0]] = self.cid

        diag_idx = comp_idx[p3 == 1]
        if len(diag_idx) == 0: return ypred

        # M4
        x_diag = x[diag_idx]
        x_f4 = self._filter_features(x_diag, self.select_indices_model4)
        x_s4 = self.scaler4.transform(x_f4)
        p4 = self.model4.predict(x_s4)

        for i, val in enumerate(p4):
            ypred[diag_idx[i]] = self.cdl if val == 0 else self.cdr

        return ypred


def visualize_model(model_clf, scaler, select_indices, x_data, y_true, title, label_map, feature_names):
    if len(x_data) == 0: return

    x_filtered = x_data[:, select_indices]
    x_scaled = scaler.transform(x_filtered)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    w = model_clf.coef_[0]
    b = model_clf.intercept_[0]
    used_names = [feature_names[i] for i in select_indices]
    eq = " + ".join([f"({w[i]:.2f}*{used_names[i]})" for i in range(len(w))])
    print(f"\n[{title}] Hyperplane: {eq} + {b:.2f} = 0")

    # 차원에 따른 시각화 처리
    if x_filtered.shape[1] >= 3:
        # 3차원 이상: PCA로 3차원 축소 후 시각화
        pca = PCA(n_components=3)
        x_vis = pca.fit_transform(x_scaled)

        # 가중치 변환
        w_pca = w @ pca.components_.T
        b_pca = b + np.dot(w, pca.mean_)

        # 격자 생성
        x_min, x_max = x_vis[:, 0].min() - 1, x_vis[:, 0].max() + 1
        y_min, y_max = x_vis[:, 1].min() - 1, x_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

        if abs(w_pca[2]) > 0.001:
            z = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
            ax.plot_surface(xx, yy, z, alpha=0.2, color='gray')

    else:
        # 2차원 이하: 3차원 공간의 바닥(z=0)에 투영하거나 수직 평면으로 표현
        # 여기서는 x, y축을 피쳐로 쓰고 z축은 0으로 두되, 결정 경계는 수직 평면으로 그림
        x_vis = np.hstack([x_scaled, np.zeros((len(x_scaled), 1))])

        x_min, x_max = x_vis[:, 0].min() - 1, x_vis[:, 0].max() + 1
        y_min, y_max = x_vis[:, 1].min() - 1, x_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

        # w[0]x + w[1]y + b = 0 => z에 상관없는 수직 평면
        # 하지만 plot_surface를 위해 z를 x,y의 함수로 표현하기 어려우므로(수직이어서),
        # 등고선이나 scatter만 표시하거나, 아래처럼 x, z를 mesh로 두고 y를 구함
        if abs(w[1]) > 0.001:
            zz, xx_v = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(x_min, x_max, 10))
            yy_v = -(w[0] * xx_v + b) / w[1]
            ax.plot_surface(xx_v, yy_v, zz, alpha=0.2, color='gray')

    # 데이터 산점도
    preds = model_clf.predict(x_scaled)
    colors = ['blue', 'red']

    # 범례용 이름 매핑
    inv_label = {v: k for k, v in label.items()}

    for cls_label, binary_target in label_map.items():
        idxs = np.where(y_true == cls_label)[0]
        if len(idxs) == 0: continue

        cls_name = inv_label[cls_label]
        correct = idxs[preds[idxs] == binary_target]
        wrong = idxs[preds[idxs] != binary_target]

        c = colors[binary_target]
        if len(correct) > 0:
            ax.scatter(x_vis[correct, 0], x_vis[correct, 1], x_vis[correct, 2],
                       c=c, marker='o', s=40, alpha=0.8, label=f'{cls_name}')
        if len(wrong) > 0:
            ax.scatter(x_vis[wrong, 0], x_vis[wrong, 1], x_vis[wrong, 2],
                       c=c, marker='x', s=100, label=f'{cls_name} (Wrong)')

    ax.set_title(title)
    ax.set_xlabel('Comp 1')
    ax.set_ylabel('Comp 2')
    ax.set_zlabel('Comp 3')
    ax.view_init(elev=25, azim=135)
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
    model.fit(xtrain, ytrainaug, xauglist)
    joblib.dump(model, 'mainmodel2.joblib')

    # 테스트셋 추론
    ypred = model.predict(xtest, xtestorig)

    # 1. 상세 로그 출력
    classnames = list(label.keys())
    predicted_labels = [classnames[p] for p in ypred]
    true_labels = [classnames[t] for t in ytest]

    print("\n" + "=" * 30)
    print(f"Test Set Evaluation ({len(xtest)} samples)")
    for i in range(len(ypred)):
        print(f"Sample {i + 1}: Predict={predicted_labels[i]}, True={true_labels[i]}")

    # 2. 리포트 및 Confusion Matrix
    print("\n" + "=" * 30)
    print(classification_report(ytest, ypred, target_names=classnames, zero_division=0))

    cm = confusion_matrix(ytest, ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classnames, yticklabels=classnames)
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # 3. 시각화 (M1, M2, M3, M4)
    try:
        visualize_model(model.model1, model.scaler1, model.select_indices_model1, xtrain, ytrainaug,
                        "M1: Horiz vs Rest",
                        {label['horizontal']: 0, label['vertical']: 1, label['circle']: 1,
                         label['diagonal_left']: 1, label['diagonal_right']: 1}, featurenames)

        m2_mask = (ytrainaug != label['horizontal'])
        if np.sum(m2_mask) > 0:
            visualize_model(model.model2, model.scaler2, model.select_indices_model2, xtrain[m2_mask],
                            ytrainaug[m2_mask],
                            "M2: Vert vs Complex",
                            {label['vertical']: 0, label['circle']: 1,
                             label['diagonal_left']: 1, label['diagonal_right']: 1}, featurenames)

        m3_mask = m2_mask & (ytrainaug != label['vertical'])
        if np.sum(m3_mask) > 0:
            visualize_model(model.model3, model.scaler3, model.select_indices_model3, xtrain[m3_mask],
                            ytrainaug[m3_mask],
                            "M3: Circle vs Diag",
                            {label['circle']: 0, label['diagonal_left']: 1, label['diagonal_right']: 1}, featurenames)

        m4_mask = m3_mask & (ytrainaug != label['circle'])
        if np.sum(m4_mask) > 0:
            visualize_model(model.model4, model.scaler4, model.select_indices_model4, xtrain[m4_mask],
                            ytrainaug[m4_mask],
                            "M4: Diag Left vs Right",
                            {label['diagonal_left']: 0, label['diagonal_right']: 1}, featurenames)
    except Exception as e:
        print(f"Visualization Error: {e}")