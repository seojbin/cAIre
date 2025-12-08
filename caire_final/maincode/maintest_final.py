import numpy as np
import joblib
import sys
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import datetime

current = os.path.abspath(__file__)
script_dir = os.path.dirname(current)
project_root = os.path.dirname(script_dir)

TEST_DATA_PATH = os.path.join(project_root, 'testdata')
MODEL_PATH = os.path.join(script_dir, 'mainmodel_final.joblib')

sys.path.append(project_root)

try:
    from postprocess.preprocess import label, parse
    from postprocess.feature_extractor import extractfeatures
    from maincode.maintrain_final import HybridClassifier 
except ImportError as e:
    print(f"Error: {e}")
    print("Ensure 'postprocess' folder and 'maintrain_final.py' are accessible.")
    exit()

def visualize_model_test(model_clf, scaler, select_indices, x_data, y_true, title, save_path, label_map, feature_names):
    if len(x_data) == 0: return ""
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

    ax.scatter(x_vis[:, 0], x_vis[:, 1], x_vis[:, 2], c='black', marker='o', s=40, label='Test Data')

    ax.set_title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

    return hyperplane_eq

def run_demo():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(script_dir, f"test_results_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    log_path = os.path.join(result_dir, "test_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def print_log(msg): 
        log_file.write(msg + "\n")

    print_log(f"=== AI Model Live Demo ===")
    print_log(f"Date: {timestamp}")
    print_log(f"Target Data Path: {TEST_DATA_PATH}")
    print_log(f"Model File: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return

    model = joblib.load(MODEL_PATH)

    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Data path '{TEST_DATA_PATH}' does not exist!")
        return

    
    test_files = sorted(glob.glob(os.path.join(TEST_DATA_PATH, '*.txt')))
    
    x_test = []
    valid_filenames = []

    for fpath in test_files:
        data = parse(fpath)
        if data is not None and len(data) > 0:
            x_test.append(data)
            valid_filenames.append(os.path.basename(fpath))
        else:
            print_log(f"Warning: Failed to parse {os.path.basename(fpath)}")

    x_test = np.array(x_test, dtype=object)
    
    y_true = np.zeros(len(x_test)) 

    if len(x_test) == 0:
        print("Error: No valid data found.")
        return
    print_log(f"   -> Loaded {len(x_test)} valid samples.")

    print_log(">> Extracting features...")
    try:
        x_features, feature_names = extractfeatures(x_test)
    except Exception as e:
        print(f"Feature Extraction Error: {e}")
        return
    
    y_pred = model.predict(x_features)
    
    inv_label = {v: k for k, v in label.items()}
    
    submission_path = os.path.join(result_dir, "submission.txt")
    sub_file = open(submission_path, "w", encoding="utf-8")
    
    for i, pred in enumerate(y_pred):
        pred_name = inv_label[pred]
        fname = valid_filenames[i]
        
        res_str = f"{fname}: {pred_name}"
        
        print(res_str)
        sub_file.write(res_str + "\n")
        print_log(f"Sample {i+1:02d}: {res_str}") 

    sub_file.close()

    try:
        eq1 = visualize_model_test(model.model1, model.scaler1, model.select_indices_model1, 
                        x_features, y_true, 
                        "Test M1: Circle vs Rest", os.path.join(result_dir, "M1_vis.png"),
                        {}, feature_names)
        log_file.write(f"\n{eq1}\n")
        
        eq2 = visualize_model_test(model.model2, model.scaler2, model.select_indices_model2, 
                        x_features, y_true, 
                        "Test M2: Horizontal vs Rest", os.path.join(result_dir, "M2_vis.png"),
                        {}, feature_names)
        log_file.write(f"{eq2}\n")
            
        eq3 = visualize_model_test(model.model3, model.scaler3, model.select_indices_model3, 
                        x_features, y_true, 
                        "Test M3: Vertical vs Diagonal", os.path.join(result_dir, "M3_vis.png"),
                        {}, feature_names)
        log_file.write(f"{eq3}\n")

        eq4 = visualize_model_test(model.model4, model.scaler4, model.select_indices_model4, 
                        x_features, y_true, 
                        "Test M4: Diag L vs R (Apex)", os.path.join(result_dir, "M4_vis.png"),
                        {}, feature_names)
        log_file.write(f"{eq4}\n")
                
    except Exception as e:
        print_log(f"Visualization Error: {e}")

    log_file.close()

if __name__ == "__main__":
    run_demo()