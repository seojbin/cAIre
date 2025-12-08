import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import datetime

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)

TEST_DATA_PATH = os.path.join(project_root, 'newdata')
MODEL_PATH = os.path.join(script_dir, 'mainmodel_final.joblib')

sys.path.append(project_root)

try:
    from postprocess.preprocess import load, label
    from postprocess.feature_extractor import extractfeatures
    from maincode.maintrain_final import HybridClassifier 
except ImportError as e:
    print(f"Error: {e}")
    print("Ensure 'postprocess' folder and 'maintrain_final.py' are accessible.")
    exit()

def visualize_model_test(model_clf, scaler, select_indices, x_data, y_true, title, save_path, label_map, feature_names):
    if len(x_data) == 0: return
    x_f = x_data[:, select_indices]
    x_s = scaler.transform(x_f)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    w = model_clf.coef_[0]
    b = model_clf.intercept_[0]
    eq_parts = [f"({w[i]:.2f}*{feature_names[select_indices[i]]})" for i in range(len(w))]
    hyperplane_eq = f"[{title}] Hyperplane: {' + '.join(eq_parts)} + {b:.2f} = 0"
    
    if x_f.shape[1] >= 3:
        pca = PCA(n_components=3)
        x_vis = pca.fit_transform(x_s)
        w_pca = w @ pca.components_.T
        b_pca = b + np.dot(w, pca.mean_)
        x_min, x_max = x_vis[:, 0].min(), x_vis[:, 0].max()
        y_min, y_max = x_vis[:, 1].min(), x_vis[:, 1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
        if abs(w_pca[2]) > 0.001:
            z = -(w_pca[0] * xx + w_pca[1] * yy + b_pca) / w_pca[2]
            ax.plot_surface(xx, yy, z, alpha=0.2, color='gray')
    else:
        x_vis = np.hstack([x_s, np.zeros((len(x_s), 3 - x_s.shape[1]))])

    preds = model_clf.predict(x_s)
    colors = ['blue', 'red']
    inv_label = {v: k for k, v in label.items()}
    has_label = (len(y_true) == len(preds))

    if has_label:
        for cls_label, target_val in label_map.items():
            idxs = np.where(y_true == cls_label)[0]
            if len(idxs) == 0: continue
            
            correct = idxs[preds[idxs] == target_val]
            wrong = idxs[preds[idxs] != target_val]
            c = colors[target_val] if target_val < 2 else 'green'
            lbl = inv_label[cls_label]
            
            if len(correct) > 0:
                ax.scatter(x_vis[correct, 0], x_vis[correct, 1], x_vis[correct, 2], 
                           c=c, marker='o', s=40, alpha=0.7, label=f'{lbl}')
            if len(wrong) > 0:
                ax.scatter(x_vis[wrong, 0], x_vis[wrong, 1], x_vis[wrong, 2], 
                           c=c, marker='x', s=100, label=f'{lbl} (Wrong)')
    else:
        ax.scatter(x_vis[:, 0], x_vis[:, 1], x_vis[:, 2], c='black', marker='o', s=40, label='Test Data')

    ax.set_title(title)
    plt.legend()
    plt.savefig(save_path)
    # plt.show() 

    return hyperplane_eq

def visualize_failed_samples(x_raw, y_true, y_pred, save_dir):
    inv_label = {v: k for k, v in label.items()}
    failed_indices = np.where(y_true != y_pred)[0]
    
    if len(failed_indices) == 0:
        return

    fail_dir = os.path.join(save_dir, "failed_samples")
    os.makedirs(fail_dir, exist_ok=True)

    for idx in failed_indices:
        traj = x_raw[idx]
        true_lbl = inv_label[y_true[idx]]
        pred_lbl = inv_label[y_pred[idx]]
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(traj[:,0], traj[:,1], traj[:,2], label='Trajectory', color='red')
        ax.scatter(traj[0,0], traj[0,1], traj[0,2], c='green', marker='o', s=50, label='Start')
        ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], c='blue', marker='x', s=50, label='End')
        
        ax.set_title(f"Sample {idx}: True[{true_lbl}] vs Pred[{pred_lbl}]")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        filename = f"Fail_ID{idx}_True_{true_lbl}_Pred_{pred_lbl}.png"
        plt.savefig(os.path.join(fail_dir, filename))
        plt.close()

def run_demo():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(script_dir, f"test_results_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    log_path = os.path.join(result_dir, "test_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def print_log(msg):
        log_file.write(msg + "\n")

    print_log(f"Date: {timestamp}")
    print_log(f"Target Data Path: {TEST_DATA_PATH}")
    print_log(f"Model File: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print_log(f"Error: Model file '{MODEL_PATH}' not found!")
        return

    model = joblib.load(MODEL_PATH)

    if not os.path.exists(TEST_DATA_PATH):
        print_log(f"Error: Data path '{TEST_DATA_PATH}' does not exist")
        return

    x_test, y_true = load(TEST_DATA_PATH)
    
    if len(x_test) == 0:
        print_log("Error: No data found.")
        return
    print_log(f"   -> Loaded {len(x_test)} samples.")

    print_log(">> Extracting features...")
    x_features, feature_names = extractfeatures(x_test)
    
    print_log(">> Running inference...")
    y_pred = model.predict(x_features)
    
    classnames = list(label.keys())
    inv_label = {v: k for k, v in label.items()}
    
    print_log("\n" + "="*40)
    print_log("       PREDICTION RESULTS       ")
    print_log("="*40)
    
    correct_cnt = 0
    has_label = (len(y_true) == len(y_pred))

    for i in range(len(y_pred)):
        pred_name = inv_label[y_pred[i]]
        if has_label:
            true_name = inv_label[y_true[i]]
            mark = "O" if y_pred[i] == y_true[i] else "X"
            if mark == "O": correct_cnt += 1
            print_log(f"Sample {i+1:02d}: Pred={pred_name:<15} | True={true_name:<15} [{mark}]")
        else:
            print_log(f"Sample {i+1:02d}: Pred={pred_name}")

    if has_label:
        acc = correct_cnt / len(y_pred) * 100
        print_log(f"Final Accuracy: {acc:.2f}% ({correct_cnt}/{len(y_pred)})")
        
        report = classification_report(y_true, y_pred, target_names=classnames, zero_division=0)
        print_log("\n[Classification Report]")
        print_log(report)
        
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=classnames, yticklabels=classnames)
            plt.title(f'Demo Result (Acc: {acc:.1f}%)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
            plt.close()
        except: pass
        if correct_cnt < len(y_pred):
            visualize_failed_samples(x_test, y_true, y_pred, result_dir)
    try:
        eq1 = visualize_model_test(model.model1, model.scaler1, model.select_indices_model1, x_features, y_true, 
                        "Test M1: Circle vs Rest", os.path.join(result_dir, "M1_vis.png"),
                        {label['circle']: 0, label['horizontal']: 1, label['vertical']: 1, 
                         label['diagonal_left']: 1, label['diagonal_right']: 1}, feature_names)
        log_file.write(f"\n{eq1}\n")
        
        if has_label:
            m2_mask = (y_true != label['circle'])
            if np.sum(m2_mask) > 0:
                eq2 = visualize_model_test(model.model2, model.scaler2, model.select_indices_model2, 
                                x_features[m2_mask], y_true[m2_mask], 
                                "Test M2: Horizontal vs Rest", os.path.join(result_dir, "M2_vis.png"),
                                {label['horizontal']: 0, label['vertical']: 1, 
                                 label['diagonal_left']: 1, label['diagonal_right']: 1}, feature_names)
                log_file.write(f"{eq2}\n")
                
            m3_mask = m2_mask & (y_true != label['horizontal'])
            if np.sum(m3_mask) > 0:
                eq3 = visualize_model_test(model.model3, model.scaler3, model.select_indices_model3, 
                                x_features[m3_mask], y_true[m3_mask], 
                                "Test M3: Vertical vs Diagonal", os.path.join(result_dir, "M3_vis.png"),
                                {label['vertical']: 0, label['diagonal_left']: 1, 
                                 label['diagonal_right']: 1}, feature_names)
                log_file.write(f"{eq3}\n")

            m4_mask = m3_mask & (y_true != label['vertical'])
            if np.sum(m4_mask) > 0:
                eq4 = visualize_model_test(model.model4, model.scaler4, model.select_indices_model4, 
                                x_features[m4_mask], y_true[m4_mask], 
                                "Test M4: Diag L vs R (Apex)", os.path.join(result_dir, "M4_vis.png"),
                                {label['diagonal_left']: 0, label['diagonal_right']: 1}, feature_names)
                log_file.write(f"{eq4}\n")
                
    except Exception as e:
        print_log(f"Visualization Error: {e}")

    log_file.close()

    submission_path = os.path.join(result_dir, "submission.txt")
    sub_file = open(submission_path, "w", encoding="utf-8")
    test_files = sorted([f for f in os.listdir(TEST_DATA_PATH) if f.endswith('.txt')])
    
    if len(test_files) == len(y_pred):
        for fname, pred in zip(test_files, y_pred):
            res_str = f"{fname}: {inv_label[pred]}"
            print(res_str)
            sub_file.write(res_str + "\n")
    else:
        for i, pred in enumerate(y_pred):
            res_str = f"test_{i+1}.txt: {inv_label[pred]}"
            print(res_str)
            sub_file.write(res_str + "\n")
            
    sub_file.close()

if __name__ == "__main__":
    run_demo()