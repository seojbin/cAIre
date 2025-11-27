import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# augmentdata
from postprocess.preprocess import load, label, augmentdata
from postprocess.feature_extractor import extractfeatures

def analyze_left_vs_right(df):
    # 라벨 필터링 (Diagonal Left vs Right)
    df_target = df[df['label_name'].isin(['diagonal_left', 'diagonal_right'])]
    
    group_left = df_target[df_target['label_name'] == 'diagonal_left']
    group_right = df_target[df_target['label_name'] == 'diagonal_right']
    
    results = []
    exclude_cols = ['label', 'label_name', 'source'] 
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    for feature in feature_cols:
        # 분산 0 스킵
        if group_left[feature].std() == 0 and group_right[feature].std() == 0:
            continue
         
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
    

    print(f"{'Rank':<4} {'Feature Name':<20} {'Importance(d)':<15} {'P-Value':<12} {'Mean(L)':<10} {'Mean(R)':<10}")

    
    for i in range(min(20, len(results_df))): 
        row = results_df.iloc[i]
        print(f"{i+1:<4} {row['feature']:<20} {row['cohens_d']:.4f}           {row['p_value']:.1e}    {row['mean_left']:.2f}       {row['mean_right']:.2f}")
        
    top_features = results_df.head(6)['feature'].tolist()
    return top_features

def plot_feature_distributions(df, top_features):
    df_filtered = df[df['label_name'].isin(['diagonal_left', 'diagonal_right'])]
    
    if len(df_filtered) > 500:
        df_filtered = df_filtered.sample(500, random_state=42)

    df_melt = df_filtered.melt(id_vars=['label_name'], value_vars=top_features, 
                      var_name='Feature', value_name='Value')
    
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df_melt, x='Feature', y='Value', hue='label_name', 
                palette={'diagonal_left': 'orange', 'diagonal_right': 'gold'})
    plt.title("Feature Distributions (Augmented Data): Diagonal Left vs Right")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

data_path = os.path.join(project_root, 'data')
newdata_path = os.path.join(project_root, 'newdata')

print("데이터 로드")
x_old, y_old = load(data_path)
x_new, y_new = load(newdata_path)

if len(x_new) > 0:
    x_raw = x_old + x_new
    y_raw = np.concatenate([y_old, y_new])
else:
    x_raw = x_old
    y_raw = y_old

print(f"원본 데이터 개수: {len(x_raw)}")

AUG_FACTOR = 9 
x_aug, y_aug = augmentdata(x_raw, y_raw, n=AUG_FACTOR)

X_features, feature_names = extractfeatures(x_aug)

df = pd.DataFrame(X_features, columns=feature_names)
df['label'] = y_aug
inv_label = {v: k for k, v in label.items()}
df['label_name'] = df['label'].map(inv_label)

top_features = analyze_left_vs_right(df)

# 시각화
plot_feature_distributions(df, top_features)

