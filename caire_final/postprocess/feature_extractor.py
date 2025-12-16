#안쓰는피쳐 제거 계산효율화 버전
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
        'X_iqr', 'Y_iqr', 'Z_iqr',    # 0,1,2
        'none', 'none', 'none', # 3, 4, 5 
        'clean_range_X', 'clean_range_Y', 'clean_range_Z', # 6, 7, 8
        'ratio_pca_resid',          # 9
        'none',              # 10 
        'xy_area_resid',              # 11
        'none',           # 12
        'none',            # 13 
        'apex_vec_x', 'apex_vec_y',        # 14, 15
        'apex_vec_z',            # 16
        'radius_ratio_resid',       # 17
        'pca_z_clean',       # 18
        'helix_thickness',            # 19
        'deviation_max',             # 20
        'none',                    # 21
        'none',                         # 22
        'none',            # 23
        'none',          # 24 
        'start_x_rel',      # 25
        'start_y_rel',           # 26
        'none',             # 27 
        'delta_azimuth',               #28
        'xy_diag_sum'        # 29
    ]

    for traj_raw in trajectories:
        if len(traj_raw) < 10:
            feature_list.append(np.zeros(len(feature_names)))
            continue

        # Smoothing
        traj_shape = gaussian_filter1d(traj_raw, sigma=2.0, axis=0)
        traj_detail = gaussian_filter1d(traj_raw, sigma=1.0, axis=0)
        start_point = traj_shape[0]

        # Detrending
        traj_clean = get_detrended_traj(traj_shape)
        traj_clean = traj_clean - traj_clean[0]

        # Scale
        p_min = np.percentile(traj_clean, 2, axis=0)
        p_max = np.percentile(traj_clean, 98, axis=0)
        clean_ranges = p_max - p_min
        total_scale = np.linalg.norm(clean_ranges) + 1e-6

        # Apex/ One-way
        apex_idx = get_robust_apex_idx(traj_clean, start_point)
        apex_point = traj_clean[apex_idx]
        main_vec = apex_point - start_point
        max_reach_clean = np.linalg.norm(main_vec)
        
        traj_oneway = traj_clean[:apex_idx+1]
        if len(traj_oneway) < 5: traj_oneway = traj_clean

        # PCA
        pca_shape = PCA(n_components=3)
        proj_shape = pca_shape.fit_transform(traj_shape)
        pc1_vals = proj_shape[:, 0]
        ideal_length = np.percentile(pc1_vals, 98) - np.percentile(pc1_vals, 2)

        pca_resid = PCA(n_components=3)
        pca_resid.fit(traj_clean)
        vars_resid = pca_resid.explained_variance_
        resid_spread = np.sqrt(vars_resid[1])


        # 0~2 IQR
        iqrs = iqr(traj_clean, axis=0, rng=(25, 75))
        
        # 6~8 Clean Range
        feat_ranges = clean_ranges

        # 9 Ratio PCA
        ratio_pca = vars_resid[1] / (vars_resid[0] + 1e-6)

        # 11 XY Area
        feat_xy_area = 0.0
        try:
            if len(traj_clean) > 3:
                hull = ConvexHull(traj_clean[:, :2])
                feat_xy_area = hull.volume
        except: pass

        # 14~16 Apex Vector
        feat_apex_vec = np.zeros(3)
        if max_reach_clean > 1e-3:
            feat_apex_vec = main_vec / max_reach_clean

        # 17 Radius Ratio
        feat_radius_ratio = 0.0
        try:
            centroid = np.mean(traj_clean, axis=0)
            dists = np.linalg.norm(traj_clean - centroid, axis=1)
            r_min = np.min(dists)
            r_max = np.max(dists)
            if r_max > 1e-6:
                feat_radius_ratio = r_min / r_max
        except: pass

        # 18 PCA Z
        feat_pca_z = abs(pca_shape.components_[2][2])

        # 19 Helix Thickness
        feat_helix_thick = 0.0
        if ideal_length > 1e-6:
            feat_helix_thick = resid_spread / ideal_length

        # 20 Deviation Max
        feat_dev_max = 0.0
        if max_reach_clean > 1e-3:
            line_unit = main_vec / max_reach_clean
            vecs = traj_oneway - start_point
            cross = np.cross(vecs, line_unit)
            feat_dev_max = np.max(np.linalg.norm(cross, axis=1)) / max_reach_clean

        # 25, 26 Start Position
        feat_start_x_rel = 0.0
        feat_start_y_rel = 0.0
        try:
            centroid = np.mean(traj_clean, axis=0)
            diff = -centroid
            if max_reach_clean > 1e-6:
                feat_start_x_rel = diff[0] / max_reach_clean
                feat_start_y_rel = diff[1] / max_reach_clean
        except: pass
        
        # 28 clockwork azimuth
        feat_delta_azimuth = 0.0
        try:
            theta_start = np.arctan2(start_point[1], start_point[0])
            theta_end = np.arctan2(apex_point[1], apex_point[0])

            diff = theta_end - theta_start
            
            feat_delta_azimuth = np.arctan2(np.sin(diff), np.cos(diff))
        except: pass
        # 29 xy_diag
        feat_xy_diag_sum = np.sqrt(feat_ranges[0]**2 + feat_ranges[1]**2)
        features = np.concatenate([
            iqrs,                                    # 0, 1, 2
            [None, None, None], # 3, 4, 5
            feat_ranges,                             # 6, 7, 8
            [ratio_pca],                       # 9
            [None],                       # 10
            [feat_xy_area],                          # 11
            [None, None],       # 12, 13
            feat_apex_vec,                # 14, 15, 16
            [feat_radius_ratio],                # 17
            [feat_pca_z],                      # 18
            [feat_helix_thick],                # 19
            [feat_dev_max],                          # 20
            [None],                 # 21
            [None, None],                   # 22, 23
            [None],          # 24
            [feat_start_x_rel, feat_start_y_rel],    # 25, 26
            [None],                      # 27
            [feat_delta_azimuth],           #28
            [feat_xy_diag_sum]                   # 29
        ])
        
        feature_list.append(features)

    return np.array(feature_list), feature_names