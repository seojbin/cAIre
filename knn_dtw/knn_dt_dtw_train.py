import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys
import os
from sklearn.tree import DecisionTreeClassifier 
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesResampler
from tensorflow.keras.preprocessing.sequence import pad_sequences

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)

sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label, augmentdata
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("전처리파일(preprocess.py, feature_extractor.py) 없음")
    exit()

class HybridClassifier:
    def __init__(self,maxneighbor1=18,maxdepthsimple=3,maxdepthcomplex=3,maxneighbor2=3,randomstate=42):
        self.randomstate = randomstate
        
        self.scaler = StandardScaler()
        self.model1 = KNeighborsClassifier(n_neighbors=maxneighbor1) 
        
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
        self.drop_indices_model1 = []
    def _filter_features(self, x):
        return np.delete(x, self.drop_indices_model1, axis=1)
    def fit(self, x, y, xorig):
        pass 

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
                    print("predict 로직에서 길이 불일치 발생")
                
        return ypred

    def getrules(self, featurenames):
        rules1 = f"KNeighborsClassifier(n_neighbors={self.model1.n_neighbors}, metric='euclidean')"
        rules3 = export_text(self.model3, feature_names=featurenames, class_names=['circle', 'diagonal'])
        rules4 = f"KNeighborsTimeSeriesClassifier(n_neighbors={self.model4.n_neighbors}, metric='{self.model4.metric}')"
        return rules1, rules3, rules4

modelpath = os.path.join(script_dir, 'knn_dt_dtw.joblib')
newdata = os.path.join(project_root, 'newdata') 

classnames = list(label.keys())

try:
    model = joblib.load(modelpath)
except IOError:
    print(f"오류: {modelpath}에 모델 없음.")
    exit()

xnew, ytrue = load(newdata) 

if len(xnew) == 0:
    print("로드안됨")
    exit()

xnewfeatures, _ = extractfeatures(xnew)

print(f"총 {len(xnewfeatures)}개 추론")

ypred = model.predict(xnewfeatures, xnew)

print("추론 결과")
predicted_labels = [classnames[p] for p in ypred]
true_labels = [classnames[t] for t in ytrue]

try:
    xnewfeatures_filtered = model._filter_features(xnewfeatures) 
    xnewfeatures_scaled = model.scaler.transform(xnewfeatures_filtered) 
    model1_train_internal_labels = model.model1._y
    model1_label_map = {0: 'horizontal', 1: 'vertical', 2: 'complex'}
    model1_train_classnames = [model1_label_map[l] for l in model1_train_internal_labels]
    dtw_train_internal_labels = model.model4._y 
    dtw_original_labels_map = model.model4.classes_ 
    dtw_train_classnames = [classnames[ dtw_original_labels_map[l] ] for l in dtw_train_internal_labels]
except Exception as e:
    print(f"DTW 설명 라벨 로드 중 오류: {e}")
    dtw_train_classnames = [] 

for i in range(len(predicted_labels)):
    print(f"샘플 {i+1}: 예측={predicted_labels[i]}, 실제={true_labels[i]}")
    try:
        x_filtered_i = model._filter_features(xnewfeatures[i].reshape(1, -1))
        x_scaled_i = model.scaler.transform(x_filtered_i)
        distances, indices = model.model1.kneighbors(x_scaled_i)
        neighbor_labels = [model1_train_classnames[idx] for idx in indices[0]]
        pred_idx = model.model1.predict(x_scaled_i)[0]
        model1_pred_name = model1_label_map.get(pred_idx, "Unknown")
        print(f"  k-NN: 예측={model1_pred_name}, 이웃={neighbor_labels}")

    except Exception as e:
        print(f"  k-NN 설명 중 오류: {e}")
    if predicted_labels[i] in ['diagonal_left', 'diagonal_right']:
        try:
            sample_orig_3d = [xnew[i]]
            sample_padded = pad_sequences(sample_orig_3d, padding='post', dtype='float32', value=np.nan)
            distances, indices = model.model4.kneighbors(sample_padded)
            
            neighbor_indices = indices[0]
            neighbor_labels = [dtw_train_classnames[idx] for idx in neighbor_indices]
            
            print(f"  DTW: {neighbor_labels}")
            
        except Exception as e:
            print(f"  DTW 설명 중 오류: {e}")

def visualize_model1(model, xtestfeatures, ytestoriginal):
    xtrain = model.model1._fit_X
    ytrain = model.model1._y 
    
    pca = PCA(n_components=2)
    xtrainpca = pca.fit_transform(xtrain)
    xtest_filtered = model._filter_features(xtestfeatures)
    xtestpca = pca.transform(model.scaler.transform(xtest_filtered))
    plt.figure(figsize=(12, 8))
    
    plt.scatter(xtrainpca[ytrain==0, 0], xtrainpca[ytrain==0, 1], c='lightblue', marker='.', alpha=0.3, label='Train: Horizontal')
    plt.scatter(xtrainpca[ytrain==1, 0], xtrainpca[ytrain==1, 1], c='lightcyan', marker='.', alpha=0.3, label='Train: Vertical')
    plt.scatter(xtrainpca[ytrain==2, 0], xtrainpca[ytrain==2, 1], c='mistyrose', marker='.', alpha=0.3, label='Train: Complex')

    colordict = {
        label['horizontal']: 'blue',    
        label['vertical']: 'cyan',      
        label['circle']: 'red',         
        label['diagonal_left']: 'orange', 
        label['diagonal_right']: 'gold'   
    }
    xtest_filtered_all = model._filter_features(xtestfeatures)
    ypredbinary = model.model1.predict(model.scaler.transform(xtest_filtered_all))
    
    for classname, classid in label.items():
        idxs = np.where(ytestoriginal == classid)[0]
        if len(idxs) == 0: continue
        
        if classid == label['horizontal']:
            target = 0
        elif classid == label['vertical']:
            target = 1
        else:
            target = 2
        
        correct_idxs = idxs[ypredbinary[idxs] == target]
        wrong_idxs = idxs[ypredbinary[idxs] != target]
        
        if len(correct_idxs) > 0:
            plt.scatter(xtestpca[correct_idxs, 0], xtestpca[correct_idxs, 1], 
                        c=colordict[classid], marker='o', s=100, edgecolor='black', 
                        label=f'Test: {classname} (Correct)')
            
        if len(wrong_idxs) > 0:
            plt.scatter(xtestpca[wrong_idxs, 0], xtestpca[wrong_idxs, 1], 
                        c=colordict[classid], marker='X', s=200, edgecolor='black', linewidth=2,
                        label=f'Test: {classname} (Wrong)')
    plt.title('Horizontal vs Vertical vs Complex (PCA)')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
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
    try:
        visualize_model1(model, xnewfeatures, ytrue)
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")
