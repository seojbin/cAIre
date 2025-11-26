import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os

# 한글 폰트
# plt.rc('font', family='Malgun Gothic')
# plt.rc('axes', unicode_minus=False)

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label
    from postprocess.feature_extractor import extractfeatures
except ImportError:
    print("전처리파일(preprocess.py, feature_extractor.py) 없음")
    exit()

def analyze_vertical_vs_diagonal(df):

    
    group_vert = df[df['label_name'] == 'circle']
    # diagonal_left(1)와 diagonal_right(2)를 합쳐서 'diagonal'로 간주
    group_diag = df[df['label_name'].str.contains('diagonal')]
    
    results = []
    feature_cols = [c for c in df.columns if c not in ['label', 'label_name']]
    
    for feature in feature_cols:
        # t-test
        # p-value가 작을수록두 집단의 분포다름
        t_stat, p_val = stats.ttest_ind(group_vert[feature], group_diag[feature], equal_var=False)
        
        mean_diff = abs(group_vert[feature].mean() - group_diag[feature].mean())
        pool_sd = np.sqrt((group_vert[feature].std()**2 + group_diag[feature].std()**2) / 2)
        cohens_d = mean_diff / pool_sd if pool_sd != 0 else 0
        
        results.append({
            'feature': feature,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'mean_vert': group_vert[feature].mean(),
            'mean_diag': group_diag[feature].mean()
        })
        
    # 순위
    results_df = pd.DataFrame(results).sort_values(by='cohens_d', ascending=False)
    
    print("\nVertical과 Diagonal을 잘 구분하는 특성")
    print("-" * 60)
    print(f"{'순위':<4} {'특성 이름':<15} {'중요도(Effect Size)':<20} {'P-Value':<15}")
    print("-" * 60)
    
    for i in range(len(results_df)):
        row = results_df.iloc[i]
        importance = row['cohens_d']
        print(f"{i+1:<4} {row['feature']:<15} {importance:.4f}               {row['p_value']:.4e}")
        
    top_features = results_df.head(5)['feature'].tolist()
    return top_features

def plot_feature_distributions(df, top_features):

    df_melt = df.melt(id_vars=['label_name'], value_vars=top_features, 
                      var_name='Feature', value_name='Value')
    
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df_melt, x='Feature', y='Value', hue='label_name')
    plt.title("Top Discriminative Features Distribution")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() # [Image of feature distribution boxplot]

data_path = os.path.join(project_root, 'data')
x_list, y_list = load(data_path)

if len(x_list) == 0:
    print("데이터 로드 실패")
    exit()

# 특성 추출
X_features, feature_names = extractfeatures(x_list)

df = pd.DataFrame(X_features, columns=feature_names)
df['label'] = y_list
inv_label = {v: k for k, v in label.items()}
df['label_name'] = df['label'].map(inv_label)

# 통계
top_svm_features = analyze_vertical_vs_diagonal(df)

# 시각화
plot_feature_distributions(df, top_svm_features)


print(f"   X_train_selected = X_train[:, { [feature_names.index(f) for f in top_svm_features] }]")
print("   clf = SVC(kernel='rbf', C=1.0)")
print("   clf.fit(X_train_selected, y_train)")
