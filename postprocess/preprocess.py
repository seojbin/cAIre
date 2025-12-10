import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io

label = {
    'circle': 0,
    'diagonal_left': 1,
    'diagonal_right': 2,
    'horizontal': 3,
    'vertical': 4
}
#endpoint
index = 6


def remove_spikes(traj, threshold_std=3.0):
    if len(traj) < 3: return traj

    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    mean_dist = np.mean(diffs)
    std_dist = np.std(diffs)

    limit = mean_dist + (threshold_std * std_dist) + 1e-6  # 0.0 방지
    # 첫 점은 유지 이후 점들은 거리가 limit보다 작아야 유지
    mask = [True]
    for d in diffs:
        if d < limit:
            mask.append(True)
        else:
            mask.append(False)  # 튀는 점 제거

    return traj[mask]


def smooth_trajectory(traj, window_size=3):

    if len(traj) < window_size: return traj

    df = pd.DataFrame(traj)
    smoothed = df.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed.values


def parse(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        r_lines = [line for line in lines if line.startswith('r,')]

        if not r_lines:
            return None

        cleaned_data = "".join(r_lines)
        df = pd.read_csv(io.StringIO(cleaned_data), header=None, engine='python')

        if index not in df.columns:
            return None

        df.dropna(subset=[index], inplace=True)
        if df.empty: return None

        series = df[index].astype(str).apply(lambda s: s.split('/'))

        valid_rows = series.apply(lambda x: len(x) == 3)
        series = series[valid_rows]

        if len(series) == 0: return None

        array = np.array(series.tolist(), dtype=float)
        return array

    except Exception as e:
        print(f"Error parsing {path}: {e}")
        return None


def load(base):
    otraject = []
    olabels = []

    for cname, cid in label.items():
        path = os.path.join(base, cname)
        if not os.path.isdir(path):
            print(f"Skipping {path} (Not found)")
            continue

        print(f"Loading {cname}")
        files = glob.glob(os.path.join(path, "*.txt"))

        count = 0
        for file in files:
            array = parse(file)
            if array is not None and len(array) > 0:
                otraject.append(array)
                olabels.append(cid)
                count += 1
        print(f" - {count} samples loaded.")

    return otraject, np.array(olabels)


def augment(traj, strength=1.0, scale_r=(0.9, 1.1)):
    newtraj = traj.copy()
    n_points = len(newtraj)
    # Gaussian Noise
    noise = np.random.normal(loc=0.0, scale=strength, size=newtraj.shape)
    newtraj += noise

    # Scaling
    if np.random.rand() > 0.3:
        scale = np.random.uniform(scale_r[0], scale_r[1])
        newtraj *= scale


    # Rotation
    if np.random.rand() > 0.3:
        # -15도 +15도 회전
        theta = np.radians(np.random.uniform(-15, 15))
        c, s = np.cos(theta), np.sin(theta)
        # Z축 회전
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        newtraj = np.dot(newtraj, R.T)

    # Drift
    if np.random.rand() > 0.5:
        drift_level = strength * 0.5
        drift_step = np.random.normal(loc=0.0, scale=drift_level, size=newtraj.shape)
        drift = np.cumsum(drift_step, axis=0)
        newtraj += drift

    #linear distortion
    if np.random.rand() > 0.7:
        distort_mag = strength * 20.0 
        drift_dir = np.random.normal(size=(1, 3))
        drift_dir /= (np.linalg.norm(drift_dir) + 1e-6)
        steps = np.linspace(0, 1, n_points).reshape(-1, 1)
        steps = steps ** 2
        linear_distortion = steps * (drift_dir * distort_mag)
        newtraj += linear_distortion
    #warp
    if np.random.rand() > 0.5:
        idxs = sorted(np.random.choice(len(newtraj), int(len(newtraj)*0.9), replace=False))
        newtraj = newtraj[idxs]

    return newtraj


def augmentdata(origx, origy, n=10):
    print(f"Augmenting data x{n}...")
    augx = []
    augy = []

    for traj, label in zip(origx, origy):
        augx.append(traj)
        augy.append(label)

        for _ in range(n):
            #노이즈 강도 약간씩 다르게
            newtraj = augment(traj, strength=1.0, scale_r=(0.85, 1.15))
            augx.append(newtraj)
            augy.append(label)

    return augx, np.array(augy)


def pad(list_data):
    return pad_sequences(list_data, padding='post', dtype='float32', truncating='post')


if __name__ == "__main__":
    base = './'  
    origx, origy = load(base)

    if len(origx) > 0:
        augx, augy = augmentdata(origx, origy, n=9)

        print(f"Total Samples: {len(augx)}")

        X_save = np.array(augx, dtype=object)

        np.savez('processed_data.npz', X=X_save, y=augy)
        print("Data saved to processed_data.npz")
    else:
        print("No data found.")