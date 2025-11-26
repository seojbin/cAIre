import numpy as np
import pandas as pd
import sys
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label, augmentdata
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("오류: 전처리 파일(postprocess 패키지) 없음")
    exit()

#동일모델 정의
class HybridClassifierRobustness:
    def __init__(self, maxneighbor2=3, randomstate=42):
        self.randomstate = randomstate
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.model1 = SVC(kernel='linear', random_state=self.randomstate)
        self.model2 = SVC(kernel='linear', random_state=self.randomstate)
        self.model3 = KNeighborsTimeSeriesClassifier(n_neighbors=maxneighbor2, metric='dtw')
        self.cid = label['circle']
        self.cdl = label['diagonal_left']
        self.cdr = label['diagonal_right']
        self.cho = label['horizontal']
        self.cve = label['vertical']
        self.drop_indices_model1 = [3,4,5,9,10,12,13]
        self.drop_indices_model2 = [0,2,3, 4, 5, 6, 8, 10,13]

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
        ytrain2 = np.where(ytrain2_orig == self.cid, 0, 1)
        
        if len(xtrain2) > 0:
            x_filtered2 = self._filter_features(xtrain2, self.drop_indices_model2)
            self.scaler2.fit(x_filtered2)
            x_scaled2 = self.scaler2.transform(x_filtered2)
            self.model2.fit(x_scaled2, ytrain2)
        
        diagonallabels = [self.cdl, self.cdr]
        diagonalmask = np.isin(y, diagonallabels)
        xtrain3 = [xorig[i] for i in range(len(xorig)) if diagonalmask[i]]
        
        if len(xtrain3) > 0:
            xtrain3 = pad_sequences(xtrain3, padding='post', dtype='float32', value=np.nan)
            ytrain3 = y[diagonalmask]
            self.model3.fit(xtrain3, ytrain3)

    def predict(self, x, xorig):
        x_filtered1 = self._filter_features(x, self.drop_indices_model1)
        x_scaled1 = self.scaler1.transform(x_filtered1)
        ypred1 = self.model1.predict(x_scaled1)
        ypred = np.zeros(len(x), dtype=int)
        ypred[ypred1 == 0] = self.cho
        ypred[ypred1 == 1] = self.cve
        complexmask = (ypred1 == 2)
        xtestcomplex = x[complexmask]
        complex_indices = np.where(complexmask)[0]
        
        if len(complex_indices) > 0:
            x_filtered2 = self._filter_features(xtestcomplex, self.drop_indices_model2)
            x_scaled2 = self.scaler2.transform(x_filtered2)
            ypred2 = self.model2.predict(x_scaled2)
            circle_mask = (ypred2 == 0)
            circle_indices = complex_indices[circle_mask]
            ypred[circle_indices] = self.cid
            diag_mask = (ypred2 == 1)
            diag_indices = complex_indices[diag_mask]
            if len(diag_indices) > 0:
                xtestdiag = [xorig[i] for i in diag_indices]
                xtestdiag = pad_sequences(xtestdiag, padding='post', dtype='float32', value=np.nan)
                ypred3 = self.model3.predict(xtestdiag)
                ypred[diag_indices] = ypred3
        return ypred

    def diagnose(self, x, xorig, y_true):
        x_filtered1 = self._filter_features(x, self.drop_indices_model1)
        x_scaled1 = self.scaler1.transform(x_filtered1)
        ypred1 = self.model1.predict(x_scaled1)
        y_true_m1 = 2
        if y_true == self.cho: y_true_m1 = 0
        elif y_true == self.cve: y_true_m1 = 1
        if ypred1[0] != y_true_m1: return "Model 1 (H vs V vs Complex)"
        if ypred1[0] != 2: return None
        x_filtered2 = self._filter_features(x, self.drop_indices_model2)
        x_scaled2 = self.scaler2.transform(x_filtered2)
        ypred2 = self.model2.predict(x_scaled2)
        y_true_m2 = 0 if y_true == self.cid else 1
        if ypred2[0] != y_true_m2: return "Model 2 (Circle vs Diagonal)"
        if ypred2[0] == 0: return None
        xtestdiag = pad_sequences([xorig], padding='post', dtype='float32', value=np.nan)
        ypred3 = self.model3.predict(xtestdiag)
        if ypred3[0] != y_true: return "Model 3 (Left vs Right)"
        return None

ITERATIONS = 50       # 반복 횟수
TEST_SIZE = 0.2        # 검증 비율
TRAIN_AUG_N = 9        # 학습 데이터 증강 배수
VAL_STRESS_N = 3       # 검증 데이터 가할 노이즈 배수

data_path = os.path.join(project_root, 'data')
newdata_path = os.path.join(project_root, 'newdata')

x_old, y_old = load(data_path)
x_new, y_new = load(newdata_path)

if len(x_new) > 0:
    x_total_raw = x_old + x_new
    y_total = np.concatenate([y_old, y_new])
else:
    x_total_raw = x_old
    y_total = y_old

indices = np.arange(len(y_total))
print(f"Total: {len(y_total)} (Classes: {np.unique(y_total)})")

sss = StratifiedShuffleSplit(n_splits=ITERATIONS, test_size=TEST_SIZE, random_state=None)
error_logs = []

for i, (train_idx, val_idx) in enumerate(sss.split(indices, y_total)):
    
    x_train_raw = [x_total_raw[k] for k in train_idx]
    y_train = y_total[train_idx]
    
    x_train_aug_raw, y_train_aug = augmentdata(x_train_raw, y_train, n=TRAIN_AUG_N)
    x_train_features, _ = extractfeatures(x_train_aug_raw)
    
    model = HybridClassifierRobustness(randomstate=42)
    model.fit(x_train_features, y_train_aug, x_train_aug_raw)
    
# 검증 데이터를 하나씩 노이즈 섞어 테스트
    
    loop_fail_count = 0
    
    for real_idx in val_idx:
        single_x_raw = [x_total_raw[real_idx]]
        single_y = np.array([y_total[real_idx]])
        
        # 노이즈 추가
        x_val_stress, y_val_stress = augmentdata(single_x_raw, single_y, n=VAL_STRESS_N)
        
        # 피쳐 추출
        x_val_stress_features, _ = extractfeatures(x_val_stress)
        
        # 예측
        preds = model.predict(x_val_stress_features, x_val_stress)
        
        # 1개라도 틀리면 실패
        if np.any(preds != y_val_stress):
            loop_fail_count += 1
            
            # 틀린 로그 기록
            wrong_locs = np.where(preds != y_val_stress)[0]
            
            err_loc = wrong_locs[0]
            failed_stage = model.diagnose(
                x_val_stress_features[err_loc].reshape(1, -1),
                x_val_stress[err_loc],
                y_val_stress[err_loc]
            )
            
            error_logs.append({
                'Iteration': i + 1,
                'Original_Index': real_idx,
                'True_Label': list(label.keys())[list(label.values()).index(y_val_stress[0])],
                'Failed_At': failed_stage,
                'Stress_Level': f"Failed on noisy copy"
            })

    # 루프별 정확도
    val_acc = 1.0 - (loop_fail_count / len(val_idx))
    sys.stdout.flush()



if not error_logs:
    print(f"모든 테스트를 100% 통과")
    print("모델 강건")
else:
    df_errors = pd.DataFrame(error_logs)
    
    print(f"\n테스트 실패 발생")
    
    print("\n노이즈에 가장 취약한 데이터 (Top 5)")
    top_errors = df_errors['Original_Index'].value_counts().head(5)
    for idx, count in top_errors.items():
        true_lbl = df_errors[df_errors['Original_Index'] == idx]['True_Label'].iloc[0]
        most_fail_stage = df_errors[df_errors['Original_Index'] == idx]['Failed_At'].mode()[0]
        print(f" - Index {idx} ({true_lbl}): {count}회 실패 (주 원인: {most_fail_stage})")
        
    print("\n>>>실패 원인 분포")
    print(df_errors['Failed_At'].value_counts())