# dl_robust_eval.py
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

from postprocess.featurizer_dl import load_trajectory_file, resample_traj, normalize_traj, compute_features
from postprocess.augment_dl import compose_augment

def load_raw_paths(root_dir):
    root = Path(root_dir)
    label_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    paths, labels = [], []
    for label_idx, lbl in enumerate(label_names):
        for fpath in (root / lbl).glob('*.txt'):
            if fpath.name.startswith('s'):
                continue
            paths.append(fpath)
            labels.append(label_idx)
    return paths, np.array(labels), label_names

def build_raw_dataset(data_root, newdata_root=None, target_len=100):
    paths, labels, label_names = load_raw_paths(data_root)
    if newdata_root is not None:
        paths2, labels2, _ = load_raw_paths(newdata_root)
        paths += paths2
        labels = np.concatenate([labels, labels2])
    return paths, labels, label_names

def traj_to_feat(path, target_len=100, augment_cfg=None):
    traj = load_trajectory_file(path)
    traj = resample_traj(traj, target_len=target_len)
    traj = normalize_traj(traj)
    if augment_cfg is not None:
        traj = compose_augment(traj, augment_cfg)
        traj = normalize_traj(traj)  # 증강 후 다시 정규화
    feats = compute_features(traj)
    return feats

def main():
    data_root = 'data'
    newdata_root = 'newdata'
    model_path = 'models/mlp_traj.h5'

    paths, labels, label_names = build_raw_dataset(data_root, newdata_root)
    paths = np.array(paths)
    # train/val split on file-level
    train_idx, val_idx = train_test_split(
        np.arange(len(paths)), test_size=0.2,
        stratify=labels, random_state=42
    )

    # train용 feature (증강 없이)
    X_train = np.stack([traj_to_feat(p) for p in paths[train_idx]], axis=0)
    y_train = labels[train_idx]
    X_val_clean = np.stack([traj_to_feat(p) for p in paths[val_idx]], axis=0)
    y_val = labels[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val_clean = scaler.transform(X_val_clean)

    model = load_model(model_path)

    # clean 성능
    y_train_pred = np.argmax(model.predict(X_train), axis=1)
    y_val_pred_clean = np.argmax(model.predict(X_val_clean), axis=1)
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc_clean = accuracy_score(y_val, y_val_pred_clean)
    print(f"[CLEAN] Train acc={train_acc:.3f}, Val acc={val_acc_clean:.3f}")

    # 루프형 Robustness 평가
    N_LOOP = 10
    augment_cfg = {
        'gaussian': 0.01,
        'drift': 0.03,
        'bias': 0.02,
        'tremor': 0.01,
    }

    val_acc_list = []
    for i in range(N_LOOP):
        X_val_aug = np.stack(
            [traj_to_feat(p, augment_cfg=augment_cfg) for p in paths[val_idx]],
            axis=0
        )
        X_val_aug = scaler.transform(X_val_aug)

        y_val_pred_aug = np.argmax(model.predict(X_val_aug), axis=1)
        acc = accuracy_score(y_val, y_val_pred_aug)
        val_acc_list.append(acc)
        print(f"[LOOP {i+1}] Val acc with augment={acc:.3f}")

    print("Augmented val acc mean:", np.mean(val_acc_list),
          "std:", np.std(val_acc_list))

if __name__ == "__main__":
    main()
