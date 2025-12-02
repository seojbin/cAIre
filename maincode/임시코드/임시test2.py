import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("전처리 파일(preprocess.py, feature_extractor.py) 없음")
    exit()


# --- Class Definition (Keeping the 4-model structure of 임시test2) ---
class HybridClassifier:
    def __init__(self, maxdepthsimple=3, maxdepthcomplex=3, maxneighbor2=3, randomstate=42):
        self.randomstate = randomstate
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler()

        self.model1 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced')
        self.model2 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced')
        self.model3 = SVC(kernel='linear', random_state=self.randomstate, class_weight='balanced')
        self.model4 = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric='dtw')

        self.cid = label['circle']
        self.cdl = label['diagonal_left']
        self.cdr = label['diagonal_right']
        self.cho = label['horizontal']
        self.cve = label['vertical']

        self.select_indices_model1 = [1, 2, 6, 7, 8, 11]
        self.select_indices_model2 = [1, 2, 6, 7, 8, 11]
        self.select_indices_model3 = [1, 7, 9, 11, 12]

    def _filter_features(self, x, indices):
        return x[:, indices]

    def fit(self, x, y, xorig_aug, xorig_clean, y_clean):
        pass

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

    def get_knn_neighbors(self, x_single_seq):
        """Helper to get neighbor labels for a single sequence"""
        train_maxlen = self.model4._X_fit.shape[1]
        seq_padded = pad_sequences([x_single_seq], maxlen=train_maxlen, padding='post', dtype='float32', value=np.nan)
        dists, idxs = self.model4.kneighbors(seq_padded)
        neighbor_indices = idxs[0]
        neighbor_labels = self.model4._y[neighbor_indices]
        return neighbor_labels

    def getrules(self, featurenames):
        rules1 = "Model 1: Linear SVM (H vs V vs Complex)"
        rules2 = "Model 2: Linear SVM (Circle vs Diagonals)"
        rules3 = f"KNN-DTW (n={self.model4.n_neighbors})"
        return rules1, rules2, rules3


modelpath = os.path.join(script_dir, 'mainmodel3.joblib')
newdata = os.path.join(project_root, 'newdata')
classnames = list(label.keys())
# Reverse label mapping for finding class name from index
inv_label = {v: k for k, v in label.items()}

try:
    model = joblib.load(modelpath)
except Exception as e:
    print(f"오류: {modelpath}에 모델 없음. ({e})")
    exit()

xnew, ytrue = load(newdata)

if len(xnew) == 0:
    print("로드안됨")
    exit()

xnewfeatures, feature_names = extractfeatures(xnew)

print(f"총 {len(xnewfeatures)}개 새 데이터 추론")

ypred = model.predict(xnewfeatures, xnew)

print("추론 결과")
predicted_labels = [classnames[p] for p in ypred]
true_labels = [classnames[t] for t in ytrue]

# Prediction Loop with KNN Details
for i in range(len(predicted_labels)):
    pred_idx = ypred[i]
    pred_name = predicted_labels[i]
    true_name = true_labels[i]

    extra_info = ""
    # If prediction is diagonal, it came from KNN. Show neighbors.
    if pred_idx in [label['diagonal_left'], label['diagonal_right']]:
        try:
            neighbors = model.get_knn_neighbors(xnew[i])
            neighbor_names = [inv_label.get(n, str(n)) for n in neighbors]
            neighbor_str = ", ".join(neighbor_names)
            extra_info = f" (Neighbors: {neighbor_str})"
        except:
            pass

    print(f"샘플 {i + 1}: 예측={pred_name}{extra_info}, 실제={true_name}")


# Visualization Function (SVM Hyperplanes)
def visualize_model_svm(model_clf, scaler, select_indices, x_data, y_true, title, label_map, feature_names):
    if len(x_data) == 0: return

    x_filtered = x_data[:, select_indices]
    x_scaled = scaler.transform(x_filtered)

    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x_scaled)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    w = model_clf.coef_[0]
    b = model_clf.intercept_[0]

    w_pca = w @ pca.components_.T
    b_pca = b + np.dot(w, pca.mean_)

    x_min, x_max = x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1
    y_min, y_max = x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1
    z_min, z_max = x_pca[:, 2].min() - 1, x_pca[:, 2].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

    if abs(w_pca[2]) > 0.001:
        z_plane = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
        mask = (z_plane >= z_min) & (z_plane <= z_max)
        if np.any(mask):
            ax.plot_surface(xx, yy, z_plane, alpha=0.3, color='gray')
            ax.plot([], [], [], color='gray', alpha=0.5, label='Decision Boundary')

    preds = model_clf.predict(x_scaled)
    colors = ['blue', 'red']  # Binary mapping

    # Mapping back for legend
    inv_map = {v: k for k, v in label.items()}

    for cls_label, binary_target in label_map.items():
        idxs = np.where(y_true == cls_label)[0]
        if len(idxs) == 0: continue

        correct = idxs[preds[idxs] == binary_target]
        wrong = idxs[preds[idxs] != binary_target]

        c = colors[binary_target]
        cls_name = inv_map[cls_label]

        if len(correct) > 0:
            ax.scatter(x_pca[correct, 0], x_pca[correct, 1], x_pca[correct, 2],
                       c=c, marker='o', s=60, edgecolor='black', alpha=0.9,
                       label=f'{cls_name}')
        if len(wrong) > 0:
            ax.scatter(x_pca[wrong, 0], x_pca[wrong, 1], x_pca[wrong, 2],
                       c=c, marker='X', s=100, edgecolor='black', linewidth=2,
                       label=f'{cls_name} (Wrong)')

    ax.set_title(title, fontsize=15)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.legend()
    ax.view_init(elev=25, azim=135)  # Match maintest view
    plt.tight_layout()
    plt.show()


if len(ytrue) > 0:
    print(classification_report(ytrue, ypred, target_names=classnames, zero_division=0))
    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classnames, yticklabels=classnames)
    plt.title('Data Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Visualizations
    try:
        # M1: Horizontal vs Rest
        visualize_model_svm(model.model1, model.scaler1, model.select_indices_model1, xnewfeatures, ytrue,
                            "M1: Horiz vs Rest",
                            {label['horizontal']: 0, label['vertical']: 1, label['circle']: 1,
                             label['diagonal_left']: 1, label['diagonal_right']: 1}, feature_names)

        # M2: Vertical vs Complex (Only for those not Horizontal)
        m2_mask = (ytrue != label['horizontal'])
        if np.sum(m2_mask) > 0:
            visualize_model_svm(model.model2, model.scaler2, model.select_indices_model2,
                                xnewfeatures[m2_mask], ytrue[m2_mask],
                                "M2: Vert vs Complex",
                                {label['vertical']: 0, label['circle']: 1,
                                 label['diagonal_left']: 1, label['diagonal_right']: 1}, feature_names)

        # M3: Circle vs Diagonal (Only for Complex)
        m3_mask = m2_mask & (ytrue != label['vertical'])
        if np.sum(m3_mask) > 0:
            visualize_model_svm(model.model3, model.scaler3, model.select_indices_model3,
                                xnewfeatures[m3_mask], ytrue[m3_mask],
                                "M3: Circle vs Diag",
                                {label['circle']: 0, label['diagonal_left']: 1, label['diagonal_right']: 1},
                                feature_names)

        # KNN Visualization Removed as requested

    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")