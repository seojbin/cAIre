# postprocess/augment_dl.py
import numpy as np

def add_gaussian_noise(traj, sigma=0.01):
    noise = np.random.normal(0.0, sigma, size=traj.shape)
    return traj + noise

def add_drift(traj, max_drift=0.05):
    T = traj.shape[0]
    t = np.linspace(0, 1, T)[:, None]
    coeff = np.random.uniform(-max_drift, max_drift, size=(1, 3))
    drift = coeff * t
    return traj + drift

def add_bias(traj, max_bias=0.05):
    bias = np.random.uniform(-max_bias, max_bias, size=(1, 3))
    return traj + bias

def add_tremor(traj, max_amp=0.02, freq_range=(0.5, 2.0)):
    T = traj.shape[0]
    t = np.linspace(0, 1, T)
    freq = np.random.uniform(*freq_range)
    phase = np.random.uniform(0, 2 * np.pi, size=(1, 3))
    amp = np.random.uniform(0, max_amp, size=(1, 3))
    tremor = amp * np.sin(2 * np.pi * freq * t[:, None] + phase)
    return traj + tremor

def compose_augment(traj, cfg):
    out = traj.copy()
    if cfg.get('gaussian', 0) > 0:
        out = add_gaussian_noise(out, sigma=cfg['gaussian'])
    if cfg.get('drift', 0) > 0:
        out = add_drift(out, max_drift=cfg['drift'])
    if cfg.get('bias', 0) > 0:
        out = add_bias(out, max_bias=cfg['bias'])
    if cfg.get('tremor', 0) > 0:
        out = add_tremor(out, max_amp=cfg['tremor'])
    return out

def augment_data(traj, n=1):

    augmented_list = []
    for _ in range(n):
        # 랜덤하게 설정을 섞어서 증강
        cfg = {
            'gaussian': np.random.uniform(0.005, 0.02) if np.random.rand() > 0.3 else 0,
            'drift': np.random.uniform(0.01, 0.05) if np.random.rand() > 0.5 else 0,
            'bias': np.random.uniform(0.01, 0.05) if np.random.rand() > 0.5 else 0,
            'tremor': np.random.uniform(0.01, 0.03) if np.random.rand() > 0.5 else 0
        }
        aug_traj = compose_augment(traj, cfg)
        augmented_list.append(aug_traj)
    return augmented_list
