import numpy as np
import pandas as pd
import sys
import os
import contextlib
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# 환경 설정
# ==========================================
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

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ==========================================
# 파라미터 설정
# ==========================================
ITERATIONS = 10             # 총 반복 횟수
TEST_SIZE = 0.2             
TRAIN_AUG_N = 9             # 학습 데이터 증강 배수 (개수)

# [핵심 변경] 검증용 노이즈 설정
VAL_COPY_N = 10             # 검증 시 샘플당 생성할 복제 개수
NOISE_LEVELS = [20.0, 30.0, 40.0, 50.0] # 테스트할 노이즈 강도 (Sigma) 리스트
# 1.0: 기본 노이즈, 2.0: 2배, 5.0: 극한 노이즈

# ==========================================
# 커스텀 노이즈 생성 함수 (직접 구현)
# ==========================================
def custom_noise_augment(x_raw, y_raw, n_copies, noise_sigma):
    """
    기존 augmentdata 대신, 직접 노이즈 강도(sigma)를 조절하여 데이터를 증강합니다.
    """
    aug_x = []
    aug_y = []
    
    for traj, lbl in zip(x_raw, y_raw):
        # 원본 1개 추가 (선택 사항, 여기선 스트레스 테스트라 제외 가능하지만 포함함)
        aug_x.append(traj)
        aug_y.append(lbl)
        
        for _ in range(n_copies):
            # 가우시안 노이즈 생성 (Scale = noise_sigma)
            noise = np.random.normal(loc=0.0, scale=noise_sigma, size=traj.shape)
            new_traj = traj + noise
            
            aug_x.append(new_traj)
            aug_y.append(lbl)
            
    return aug_x, np.array(aug_y)

# ==========================================
# 모델 클래스
# ==========================================
class HybridClassifierRobustness:
    def __init__(self, randomstate=42):
        self.randomstate = randomstate
        self.scaler1 = StandardScaler(); self.scaler2 = StandardScaler()
        self.scaler3 = StandardScaler(); self.scaler4 = StandardScaler()
        
        self.model1 = SVC(kernel='linear', random_state=randomstate, class_weight='balanced')
        self.model2 = SVC(kernel='linear', random_state=randomstate, class_weight='balanced')
        self.model3 = SVC(kernel='linear', random_state=randomstate, class_weight='balanced')
        self.model4 = SVC(kernel='linear', random_state=randomstate, class_weight='balanced')
        
        self.cid = label['circle']; self.cdl = label['diagonal_left']; self.cdr = label['diagonal_right']
        self.cho = label['horizontal']; self.cve = label['vertical']
        
        # M1: Circle vs Rest
        self.select_indices_model1 = [9, 11, 16] 
        # M2: H vs Rest
        self.select_indices_model2 = [1, 2, 6, 7, 8]
        # M3: V vs Diag
        self.select_indices_model3 = [1, 2, 6, 7, 8]
        # M4: L vs R
        self.select_indices_model4 = [14, 15]

    def _filter_features(self, x, indices): return x[:, indices]

    def fit(self, x, y):
        # M1
        x_f1 = self._filter_features(x, self.select_indices_model1)
        self.scaler1.fit(x_f1)
        self.model1.fit(self.scaler1.transform(x_f1), np.where(y == self.cid, 0, 1))
        # M2
        mask_m2 = (y != self.cid); x_m2, y_m2 = x[mask_m2], y[mask_m2]
        if len(x_m2) > 0:
            x_f2 = self._filter_features(x_m2, self.select_indices_model2)
            self.scaler2.fit(x_f2)
            self.model2.fit(self.scaler2.transform(x_f2), np.where(y_m2 == self.cho, 0, 1))
            # M3
            mask_m3 = (y_m2 != self.cho); x_m3, y_m3 = x_m2[mask_m3], y_m2[mask_m3]
            if len(x_m3) > 0:
                x_f3 = self._filter_features(x_m3, self.select_indices_model3)
                self.scaler3.fit(x_f3)
                self.model3.fit(self.scaler3.transform(x_f3), np.where(y_m3 == self.cve, 0, 1))
                # M4
                mask_m4 = (y_m3 != self.cve); x_m4, y_m4 = x_m3[mask_m4], y_m3[mask_m4]
                if len(x_m4) > 0:
                    x_f4 = self._filter_features(x_m4, self.select_indices_model4)
                    self.scaler4.fit(x_f4)
                    self.model4.fit(self.scaler4.transform(x_f4), np.where(y_m4 == self.cdl, 0, 1))

    def predict(self, x):
        ypred = np.zeros(len(x), dtype=int)
        # M1
        x_f1 = self._filter_features(x, self.select_indices_model1)
        p1 = self.model1.predict(self.scaler1.transform(x_f1))
        ypred[p1 == 0] = self.cid
        rest_idx = np.where(p1 == 1)[0]
        if len(rest_idx) == 0: return ypred
        # M2
        x_rest = x[rest_idx]; x_f2 = self._filter_features(x_rest, self.select_indices_model2)
        p2 = self.model2.predict(self.scaler2.transform(x_f2))
        ypred[rest_idx[p2 == 0]] = self.cho
        rest_idx2 = rest_idx[p2 == 1]
        if len(rest_idx2) == 0: return ypred
        # M3
        x_rest2 = x[rest_idx2]; x_f3 = self._filter_features(x_rest2, self.select_indices_model3)
        p3 = self.model3.predict(self.scaler3.transform(x_f3))
        ypred[rest_idx2[p3 == 0]] = self.cve
        diag_idx = rest_idx2[p3 == 1]
        if len(diag_idx) == 0: return ypred
        # M4
        x_diag = x[diag_idx]; x_f4 = self._filter_features(x_diag, self.select_indices_model4)
        p4 = self.model4.predict(self.scaler4.transform(x_f4))
        for i, val in enumerate(p4): ypred[diag_idx[i]] = self.cdl if val == 0 else self.cdr
        return ypred

    def diagnose(self, x_single, y_true):
        x = x_single.reshape(1, -1)
        # M1
        p1 = self.model1.predict(self.scaler1.transform(self._filter_features(x, self.select_indices_model1)))[0]
        if y_true == self.cid: return "M1 (Missed Circle)" if p1 != 0 else None
        if p1 == 0: return "M1 (False Circle)"
        # M2
        p2 = self.model2.predict(self.scaler2.transform(self._filter_features(x, self.select_indices_model2)))[0]
        if y_true == self.cho: return "M2 (Missed Horiz)" if p2 != 0 else None
        if p2 == 0: return "M2 (False Horiz)"
        # M3
        p3 = self.model3.predict(self.scaler3.transform(self._filter_features(x, self.select_indices_model3)))[0]
        if y_true == self.cve: return "M3 (Missed Vert)" if p3 != 0 else None
        if p3 == 0: return "M3 (False Vert)"
        # M4
        p4 = self.model4.predict(self.scaler4.transform(self._filter_features(x, self.select_indices_model4)))[0]
        target = 0 if y_true == self.cdl else 1
        if p4 != target: return f"M4 ({'L->R' if target==0 else 'R->L'} Fail)"
        return "Unknown"

# ==========================================
# 메인 실행
# ==========================================
def main():
    data_path = os.path.join(project_root, 'data')
    newdata_path = os.path.join(project_root, 'newdata')

    # 데이터 로드
    with suppress_stdout():
        x_old, y_old = load(data_path)
        x_new, y_new = load(newdata_path)

    if len(x_new) > 0:
        x_total_raw = x_old + x_new
        y_total = np.concatenate([y_old, y_new])
    else:
        x_total_raw = x_old
        y_total = y_old

    print(f"\n[데이터셋] 총 {len(y_total)}개 샘플")
    print(f" -> 검증 시 노이즈 강도(Sigma) 단계: {NOISE_LEVELS}")
    print(f" -> 각 단계별 샘플 복제 수: {VAL_COPY_N}개")
    
    sss = StratifiedShuffleSplit(n_splits=ITERATIONS, test_size=TEST_SIZE, random_state=42)
    error_logs = []

    for i, (train_idx, val_idx) in enumerate(sss.split(x_total_raw, y_total)):
        
        # [조건] 1회차 고정 분할
        if i == 0:
            if len(x_new) == 0: continue
            print(f"\n[Iter 1] Special Mode: Train(data) vs Val(newdata)")
            x_train_raw = x_old
            y_train = y_old
            val_real_indices = list(range(len(x_old), len(x_total_raw)))
        else:
            x_train_raw = [x_total_raw[k] for k in train_idx]
            y_train = y_total[train_idx]
            val_real_indices = val_idx

        # 1. 학습 (기본 증강만 적용)
        with suppress_stdout():
            x_train_aug, y_train_aug = augmentdata(x_train_raw, y_train, n=TRAIN_AUG_N)
            x_train_feat, _ = extractfeatures(x_train_aug)

        model = HybridClassifierRobustness(randomstate=42)
        model.fit(x_train_feat, y_train_aug)

        print(f"[Iter {i+1}] Training Done. Starting Stress Test...")

        # 2. 검증 (노이즈 강도별 루프)
        for noise_sigma in NOISE_LEVELS:
            level_fail_count = 0
            total_tests = 0
            
            for real_idx in val_real_indices:
                single_x_raw = [x_total_raw[real_idx]]
                single_y = np.array([y_total[real_idx]])
                
                # [핵심] 커스텀 노이즈 함수 사용 (강도 조절)
                x_val_stress, y_val_stress = custom_noise_augment(
                    single_x_raw, single_y, 
                    n_copies=VAL_COPY_N, 
                    noise_sigma=noise_sigma # 강도 적용
                )
                
                # 피쳐 추출
                with suppress_stdout():
                    x_val_feat, _ = extractfeatures(x_val_stress)
                
                preds = model.predict(x_val_feat)
                total_tests += len(preds)
                
                # 오류 분석
                mismatch_indices = np.where(preds != y_val_stress)[0]
                if len(mismatch_indices) > 0:
                    level_fail_count += len(mismatch_indices)
                    
                    # 샘플당 1번만 상세 로그 (너무 많음 방지)
                    err_idx = mismatch_indices[0]
                    cause = model.diagnose(x_val_feat[err_idx], y_val_stress[err_idx])
                    
                    true_lbl = list(label.keys())[list(label.values()).index(y_val_stress[0])]
                    error_logs.append({
                        'Iter': i+1,
                        'Noise_Sigma': noise_sigma,
                        'Original_Index': real_idx,
                        'True_Label': true_lbl,
                        'Cause': cause
                    })
            
            acc = 1.0 - (level_fail_count / total_tests)
            print(f"   -> Noise {noise_sigma:.1f}: Acc {acc*100:.2f}% ({level_fail_count}/{total_tests} fails)")

    # ==========================================
    # 리포트 저장
    # ==========================================
    with open("robustness_intensity_report.txt", "w", encoding="utf-8") as f:
        f.write("Noise Intensity Robustness Report\n")
        f.write("=================================\n\n")
        
        if not error_logs:
            f.write("PERFECT SCORE across all noise levels!\n")
            print("\n모델이 모든 노이즈 단계에서 완벽하게 동작.")
        else:
            df = pd.DataFrame(error_logs)
            
            # 노이즈 레벨별 생존율 요약
            f.write("[Failure Counts by Noise Level]\n")
            f.write(df['Noise_Sigma'].value_counts().sort_index().to_string())
            f.write("\n\n")
            
            # 주요 실패 원인
            f.write("[Top Failure Causes]\n")
            f.write(df['Cause'].value_counts().to_string())
            f.write("\n\n")
            
            # 취약한 클래스
            f.write("[Most Vulnerable Classes]\n")
            f.write(df['True_Label'].value_counts().to_string())
            
            print(f"\n총 {len(df)}건의 테스트 실패가 기록.")
            print(f"'robustness_intensity_report.txt'를 확인.")

if __name__ == "__main__":
    main()