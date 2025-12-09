# postprocess/augment_dl.py
import numpy as np

def add_gaussian_noise(traj, sigma=0.01):
    noise = np.random.normal(0.0, sigma, size=traj.shape)
    return traj + noise

def add_drift(traj, max_drift=0.05):
    """
    선형 drift: t=0에서 0, t=end에서 최대 max_drift까지 축별 랜덤 계수 [file:query]
    """
    T = traj.shape[0]
    t = np.linspace(0, 1, T)[:, None]
    coeff = np.random.uniform(-max_drift, max_drift, size=(1, 3))
    drift = coeff * t
    return traj + drift

def add_bias(traj, max_bias=0.05):
    bias = np.random.uniform(-max_bias, max_bias, size=(1, 3))
    return traj + bias

def add_tremor(traj, max_amp=0.02, freq_range=(0.5, 2.0)):
    """
    저주파 떨림: 사인파 기반 노이즈 [file:query]
    """
    T = traj.shape[0]
    t = np.linspace(0, 1, T)
    freq = np.random.uniform(*freq_range)
    phase = np.random.uniform(0, 2 * np.pi, size=(1, 3))
    amp = np.random.uniform(0, max_amp, size=(1, 3))
    tremor = amp * np.sin(2 * np.pi * freq * t[:, None] + phase)
    return traj + tremor

def compose_augment(traj, cfg):
    """
    cfg: dict 예: {'gaussian':0.01,'drift':0.03,'bias':0.02,'tremor':0.01}
    """
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
