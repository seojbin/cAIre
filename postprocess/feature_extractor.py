import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from scipy.stats import iqr
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import pywt

def get_robust_apex_idx(traj, start_point):
    if len(traj) < 5: return len(traj)-1
    dists = np.linalg.norm(traj - start_point, axis=1)
    dists_smooth = gaussian_filter1d(dists, sigma=2.0)
    return np.argmax(dists_smooth)

def get_detrended_traj(traj):
    return signal.detrend(traj, axis=0, type='linear')

def extractfeatures(trajectories):
    feature_list = []
    
    feature_names = [
        'X_iqr', 'Y_iqr', 'Z_iqr',           # 0, 1, 2
        'ideal_length_pc1', 'max_reach', 'linearity_shape', # 3, 4, 5
        'clean_range_X', 'clean_range_Y', 'clean_range_Z', # 6, 7, 8
        'ratio_pca_resid',                   # 9
        'jerk_smooth',                       # 10
        'xy_area_resid',                     # 11
        'slope_xy_clean',                    # 12 
        'corr_xy_clean',                     # 13 
        'apex_vec_x', 'apex_vec_y',          # 14, 15
        'apex_vec_z',                        # 16 
        'radius_ratio_resid',                # 17
        'pca_z_clean',                       # 18 
        'helix_thickness',                   # 19
        'deviation_max',                     # 20 
        'turn_angle_sum',                    # 21 
        'ldlj',                              # 22
        'dwt_energy_detail',                 # 23
        'linearity_resid',                   # 24
        'start_x_rel',                       # 25
        'start_y_rel',                       # 26 
        'pca_1_z',                           # 27
    ]

    for traj_raw in trajectories:
        if len(traj_raw) < 10:
            feature_list.append(np.zeros(len(feature_names)))
            continue

        # 1. Smoothing
        traj_shape = gaussian_filter1d(traj_raw, sigma=2.0, axis=0)
        traj_detail = gaussian_filter1d(traj_raw, sigma=1.0, axis=0)
        start_point = traj_shape[0]

        # 2. Detrending
        traj_clean = get_detrended_traj(traj_shape)
        traj_clean = traj_clean - traj_clean[0]

        # Scale for cutoff
        p_min = np.percentile(traj_clean, 2, axis=0)
        p_max = np.percentile(traj_clean, 98, axis=0)
        clean_ranges = p_max - p_min
        total_scale = np.linalg.norm(clean_ranges) + 1e-6

        # 3. Apex & One-way
        apex_idx = get_robust_apex_idx(traj_clean, start_point)
        apex_point = traj_clean[apex_idx]
        main_vec = apex_point - start_point
        max_reach_clean = np.linalg.norm(main_vec)
        
        traj_oneway = traj_clean[:apex_idx+1]
        if len(traj_oneway) < 5: traj_oneway = traj_clean

        # 4. PCA
        pca_shape = PCA(n_components=3)
        proj_shape = pca_shape.fit_transform(traj_shape)
        pc1_vals = proj_shape[:, 0]
        ideal_length = np.percentile(pc1_vals, 98) - np.percentile(pc1_vals, 2)

        pca_resid = PCA(n_components=3)
        pca_resid.fit(traj_clean)
        vars_resid = pca_resid.explained_variance_
        resid_spread = np.sqrt(vars_resid[1])

        # ================= Feature Extraction =================

        # [0~2] IQR
        iqrs = iqr(traj_clean, axis=0, rng=(25, 75))

        # [3~5] Length & Shape Linearity
        feat_length = ideal_length
        feat_max_reach = max_reach_clean
        
        len_path_shape = np.sum(np.linalg.norm(np.diff(traj_shape, axis=0), axis=1))
        reach_shape = np.linalg.norm(traj_shape[-1] - start_point)
        if reach_shape < 1e-3: reach_shape = np.max(np.linalg.norm(traj_shape - start_point, axis=1))
        feat_linearity_shape = len_path_shape / (reach_shape + 1e-6)

        # [6~8] Clean Range
        feat_ranges = clean_ranges

        # [9] Ratio PCA (No Cutoff)
        ratio_pca = vars_resid[1] / (vars_resid[0] + 1e-6)

        # [10] Jerk
        vel = np.diff(traj_detail, axis=0)
        acc = np.diff(vel, axis=0)
        jerk = np.diff(acc, axis=0)
        actual_len_raw = np.sum(np.linalg.norm(np.diff(traj_detail, axis=0), axis=1))
        feat_jerk = np.sum(np.linalg.norm(jerk, axis=1)) / (actual_len_raw + 1e-6)

        # [11] XY Area
        feat_xy_area = 0.0
        try:
            if len(traj_clean) > 3:
                hull = ConvexHull(traj_clean[:, :2])
                feat_xy_area = hull.volume
        except: pass

        # [12, 13] Slope & Corr (MODIFIED)
        feat_slope = 0.0
        feat_corr = 0.0
        try:
            if len(traj_oneway) > 5:
                n = len(traj_oneway)
                sub = traj_oneway[int(n*0.1):int(n*0.9)]
                lr = LinearRegression()
                lr.fit(sub[:, 0].reshape(-1, 1), sub[:, 1])
                feat_slope = lr.coef_[0]
                c = np.corrcoef(sub[:, 0], sub[:, 1])[0, 1]
                if not np.isnan(c): 
                    feat_corr = abs(c) # <--- [수정됨] 절대값 적용 (직선은 1.0으로 통일)
        except: pass

        # [14~16] Apex Vector
        feat_apex_vec = np.zeros(3)
        if max_reach_clean > 1e-3:
            feat_apex_vec = main_vec / max_reach_clean

        # [17] Radius Ratio (MODIFIED)
        feat_radius_ratio = 0.0
        try:
            # 1. 무게중심(Centroid) 계산 [수정됨]
            centroid = np.mean(traj_clean, axis=0)
            
            # 2. 중심으로부터의 거리 계산 [수정됨: start_point -> centroid]
            dists = np.linalg.norm(traj_clean - centroid, axis=1)
            
            # 3. Robust Ratio 계산 (Min/Max 방식) [수정됨: mean/std -> min/max]
            # 원이면 min과 max가 비슷해서 1.0에 근접, 직선이면 0.0에 근접
            r_min = np.min(dists)
            r_max = np.max(dists)
            
            if r_max > 1e-6:
                feat_radius_ratio = r_min / r_max
        except: pass

        # [18] PCA Z
        feat_pca_z = abs(pca_shape.components_[2][2])

        # [19] Helix Thickness
        feat_helix_thick = 0.0
        if ideal_length > 1e-6:
            feat_helix_thick = resid_spread / ideal_length

        # [20] Deviation Max
        feat_dev_max = 0.0
        if max_reach_clean > 1e-3:
            line_unit = main_vec / max_reach_clean
            vecs = traj_oneway - start_point
            cross = np.cross(vecs, line_unit)
            feat_dev_max = np.max(np.linalg.norm(cross, axis=1)) / max_reach_clean

        # [21] Turn Angle Sum
        feat_turn = 0.0
        step = 5
        if len(traj_clean) > 2 * step:
            v1 = traj_clean[step:-step] - traj_clean[:-2*step]
            v2 = traj_clean[2*step:] - traj_clean[step:-step]
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            valid = (n1 > 1e-6) & (n2 > 1e-6)
            if np.sum(valid) > 0:
                dots = np.sum(v1[valid] * v2[valid], axis=1)
                c_th = np.clip(dots / (n1[valid] * n2[valid]), -1.0, 1.0)
                feat_turn = np.sum(np.arccos(c_th))

        # [22, 23] Detail
        feat_ldlj = 0.0
        feat_dwt = 0.0
        try:
            jerk_sq = np.sum(np.linalg.norm(np.diff(np.diff(vel, axis=0), axis=0), axis=1)**2)
            max_v = np.max(np.linalg.norm(vel, axis=1)) + 1e-6
            dim_jerk = (jerk_sq * (len(traj_detail)**3)) / (max_v**2 + 1e-9)
            feat_ldlj = -np.log(dim_jerk + 1e-9)
            coeffs = pywt.wavedec(traj_detail, 'db4', level=2, axis=0)
            feat_dwt = np.log(np.sum(np.square(coeffs[1])) + 1e-6)
        except: pass

        # [24] Resid Linearity
        len_resid = np.sum(np.linalg.norm(np.diff(traj_clean, axis=0), axis=1))
        if max_reach_clean < total_scale * 0.05:
            feat_linearity_resid = 0.0
        else:
            feat_linearity_resid = len_resid / (max_reach_clean + 1e-6)

        # [NEW 25, 26] Start Position Relative to Centroid
        feat_start_x_rel = 0.0
        feat_start_y_rel = 0.0
        try:
            centroid = np.mean(traj_clean, axis=0)
            diff = -centroid
            if max_reach_clean > 1e-6:
                feat_start_x_rel = diff[0] / max_reach_clean
                feat_start_y_rel = diff[1] / max_reach_clean
        except: pass
        
        # [27] PCA 1 Z
        feat_pca_1_z = abs(pca_shape.components_[0][2])

        # Combine
        features = np.concatenate([
            iqrs,                                    # 0, 1, 2
            [feat_length, feat_max_reach, feat_linearity_shape], # 3, 4, 5
            feat_ranges,                             # 6, 7, 8
            [ratio_pca],                             # 9
            [feat_jerk],                             # 10
            [feat_xy_area],                          # 11
            [feat_slope, feat_corr],                 # 12, 13
            feat_apex_vec,                           # 14, 15, 16
            [feat_radius_ratio],                     # 17
            [feat_pca_z],                            # 18
            [feat_helix_thick],                      # 19
            [feat_dev_max],                          # 20
            [feat_turn],                             # 21
            [feat_ldlj, feat_dwt],                   # 22, 23
            [feat_linearity_resid],                  # 24 
            [feat_start_x_rel, feat_start_y_rel],    # 25, 26
            [feat_pca_1_z]                           # 27
        ])
        
        feature_list.append(features)

    return np.array(feature_list), feature_names