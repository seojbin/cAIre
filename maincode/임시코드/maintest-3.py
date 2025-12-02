import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("전처리 파일 없음")
    exit()


# Class definition (Must match train code)
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
        self.select_indices_model1 = [1, 2, 6, 7, 8, 11]
        self.select_indices_model2 = [1, 2, 6, 7, 8, 11]
        self.select_indices_model3 = [1, 7, 9, 11, 12]
        self.select_indices_model4 = [14, 15]

    def _filter_features(self, x, indices):
        return x[:, indices]

    def fit(self, x, y, xorig):
        pass

    def predict(self, x, xorig):
        ypred = np.zeros(len(x), dtype=int)
        x_f1 = self._filter_features(x, self.select_indices_model1)
        x_s1 = self.scaler1.transform(x_f1)
        p1 = self.model1.predict(x_s1)
        ypred[p1 == 0] = self.cho
        rest_idx = np.where(p1 == 1)[0]
        if len(rest_idx) == 0: return ypred

        x_rest = x[rest_idx]
        x_f2 = self._filter_features(x_rest, self.select_indices_model2)
        x_s2 = self.scaler2.transform(x_f2)
        p2 = self.model2.predict(x_s2)
        ypred[rest_idx[p2 == 0]] = self.cve
        comp_idx = rest_idx[p2 == 1]
        if len(comp_idx) == 0: return ypred

        x_comp = x[comp_idx]
        x_f3 = self._filter_features(x_comp, self.select_indices_model3)
        x_s3 = self.scaler3.transform(x_f3)
        p3 = self.model3.predict(x_s3)
        ypred[comp_idx[p3 == 0]] = self.cid
        diag_idx = comp_idx[p3 == 1]
        if len(diag_idx) == 0: return ypred

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

    if x_filtered.shape[1] >= 3:
        pca = PCA(n_components=3)
        x_vis = pca.fit_transform(x_scaled)
        w_pca = w @ pca.components_.T
        b_pca = b + np.dot(w, pca.mean_)

        x_min, x_max = x_vis[:, 0].min() - 1, x_vis[:, 0].max() + 1
        y_min, y_max = x_vis[:, 1].min() - 1, x_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

        if abs(w_pca[2]) > 0.001:
            z = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
            ax.plot_surface(xx, yy, z, alpha=0.2, color='gray')
    else:
        # 2 Features (Apex) -> Vertical Plane Visualization
        x_vis = np.hstack([x_scaled, np.zeros((len(x_scaled), 1))])
        x_min, x_max = x_vis[:, 0].min() - 1, x_vis[:, 0].max() + 1

        if abs(w[1]) > 0.001:
            zz, xx_v = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(x_min, x_max, 10))
            yy_v = -(w[0] * xx_v + b) / w[1]
            ax.plot_surface(xx_v, yy_v, zz, alpha=0.2, color='gray')

    preds = model_clf.predict(x_scaled)
    colors = ['blue', 'red']
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
    ax.view_init(elev=25, azim=135)
    plt.legend()
    plt.show()


# --- Execution ---
modelpath = os.path.join(script_dir, 'mainmodel3.joblib')
newdata = os.path.join(project_root, 'newdata')
classnames = list(label.keys())

try:
    model = joblib.load(modelpath)
except Exception as e:
    print(f"로드 실패: {e}")
    exit()

xnew, ytrue = load(newdata)
if len(xnew) == 0:
    print("데이터 없음")
    exit()

xnewfeatures, feature_names = extractfeatures(xnew)

ypred = model.predict(xnewfeatures, xnew)

print(f"총 {len(xnewfeatures)}개 새 데이터 추론")
print("추론 결과")
predicted_labels = [classnames[p] for p in ypred]
true_labels = [classnames[t] for t in ytrue]

for i in range(len(ypred)):
    print(f"샘플 {i + 1}: 예측={predicted_labels[i]}, 실제={true_labels[i]}")

print("\n" + "=" * 30)
print(classification_report(ytrue, ypred, target_names=classnames, zero_division=0))

cm = confusion_matrix(ytrue, ypred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classnames, yticklabels=classnames)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Visualization
try:
    visualize_model(model.model1, model.scaler1, model.select_indices_model1, xnewfeatures, ytrue,
                    "Test M1: Horiz vs Rest",
                    {label['horizontal']: 0, label['vertical']: 1, label['circle']: 1,
                     label['diagonal_left']: 1, label['diagonal_right']: 1}, feature_names)

    m2_mask = (ytrue != label['horizontal'])
    if np.sum(m2_mask) > 0:
        visualize_model(model.model2, model.scaler2, model.select_indices_model2, xnewfeatures[m2_mask], ytrue[m2_mask],
                        "Test M2: Vert vs Complex",
                        {label['vertical']: 0, label['circle']: 1,
                         label['diagonal_left']: 1, label['diagonal_right']: 1}, feature_names)

    m3_mask = m2_mask & (ytrue != label['vertical'])
    if np.sum(m3_mask) > 0:
        visualize_model(model.model3, model.scaler3, model.select_indices_model3, xnewfeatures[m3_mask], ytrue[m3_mask],
                        "Test M3: Circle vs Diag",
                        {label['circle']: 0, label['diagonal_left']: 1, label['diagonal_right']: 1}, feature_names)

    m4_mask = m3_mask & (ytrue != label['circle'])
    if np.sum(m4_mask) > 0:
        visualize_model(model.model4, model.scaler4, model.select_indices_model4, xnewfeatures[m4_mask], ytrue[m4_mask],
                        "Test M4: Diag L vs R (Apex)",
                        {label['diagonal_left']: 0, label['diagonal_right']: 1}, feature_names)

except Exception as e:
    print(f"시각화 중 오류: {e}")