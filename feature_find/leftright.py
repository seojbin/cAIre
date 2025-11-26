import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os

# 한글 폰트 설정 (필요시 주석 해제)
# plt.rc('font', family='Malgun Gothic')
# plt.rc('axes', unicode_minus=False)

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)


from postprocess.preprocess import load, label
from postprocess.feature_extractor import extractfeatures


def analyze_left_vs_right(df):
    
    group_left = df[df['label_name'] == 'diagonal_left']
    group_right = df[df['label_name'] == 'diagonal_right']
    

    results = []
    feature_cols = [c for c in df.columns if c not in ['label', 'label_name']]
    
    for feature in feature_cols:
        # t-test (Welch's t-test)
        t_stat, p_val = stats.ttest_ind(group_left[feature], group_right[feature], equal_var=False)
        
        mean_diff = abs(group_left[feature].mean() - group_right[feature].mean())
        pool_sd = np.sqrt((group_left[feature].std()**2 + group_right[feature].std()**2) / 2)
        cohens_d = mean_diff / pool_sd if pool_sd != 0 else 0
        
        results.append({
            'feature': feature,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'mean_left': group_left[feature].mean(),
            'mean_right': group_right[feature].mean()
        })
        
    results_df = pd.DataFrame(results).sort_values(by='cohens_d', ascending=False)
    
    print("\nDiagonal Left vs Right 구분 특성")
    print("-" * 85)
    print(f"{'순위':<4} {'특성 이름':<15} {'중요도':<20} {'P-Value':<15} {'Mean(L)':<10} {'Mean(R)':<10}")
    print("-" * 85)
    
    for i in range(len(results_df)):
        row = results_df.iloc[i]
        print(f"{i+1:<4} {row['feature']:<15} {row['cohens_d']:.4f}               {row['p_value']:.4e}     {row['mean_left']:.2f}       {row['mean_right']:.2f}")
        
    top_features = results_df.head(5)['feature'].tolist()
    return top_features

def plot_feature_distributions(df, top_features):
    # 시각화를 위해 Left/Right 데이터만 필터링
    df_filtered = df[df['label_name'].isin(['diagonal_left', 'diagonal_right'])]
    
    df_melt = df_filtered.melt(id_vars=['label_name'], value_vars=top_features, 
                      var_name='Feature', value_name='Value')
    
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df_melt, x='Feature', y='Value', hue='label_name', palette={'diagonal_left': 'orange', 'diagonal_right': 'gold'})
    plt.title("Distribution: Diagonal Left vs Right")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

data_path = os.path.join(project_root, 'data')
x_list, y_list = load(data_path)

if len(x_list) == 0:
    print("데이터 로드 실패")
    exit()

X_features, feature_names = extractfeatures(x_list)

df = pd.DataFrame(X_features, columns=feature_names)
df['label'] = y_list
inv_label = {v: k for k, v in label.items()}
df['label_name'] = df['label'].map(inv_label)

# 분석 실행
top_features = analyze_left_vs_right(df)

# 시각화 실행
plot_feature_distributions(df, top_features)

