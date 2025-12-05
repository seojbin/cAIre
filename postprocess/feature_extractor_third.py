import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from scipy.stats import iqr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import pywt 

def get_robust_apex(traj, start_point):
    dists = np.linalg.norm(traj - start_point, axis=1)
    if len(traj) < 10:
        return traj[np.argmax(dists)], np.argmax(dists)
    threshold = np.percentile(dists, 85)
    candidates = traj[dists >= threshold]
    if len(candidates) == 0:
        return traj[np.argmax(dists)], np.argmax(dists)
    geo_median = np.median(candidates, axis=0)
    dists_from_median = np.linalg.norm(candidates - geo_median, axis=1)
    limit_dist = np.median(dists_from_median) * 2.0 + 1e-6
    clean_candidates = candidates[dists_from_median <= limit_dist]
    if len(clean_candidates) == 0: clean_candidates = candidates
    robust_apex = np.mean(clean_candidates, axis=0)
    robust_idx = np.argmin(np.linalg.norm(traj - robust_apex, axis=1))
    return robust_apex, robust_idx

def get_one_way_end_idx(traj, start_point):
    dists = np.linalg.norm(traj - start_point, axis=1)
    # Peak 탐색도 강력하게 스무딩된 데이터로 수행 (노이즈 무시)
    dists_smooth = gaussian_filter1d(dists, sigma=2.0)
    global_max_dist = np.max(dists_smooth)
    min_reach_threshold = global_max_dist * 0.5
    peaks, _ = find_peaks(dists_smooth, height=min_reach_threshold, distance=10)
    if len(peaks) == 0: return np.argmax(dists)
    first_peak_idx = peaks[0]
    post_peak_dists = dists_smooth[first_peak_idx:]
    valleys, _ = find_peaks(-post_peak_dists, distance=10)
    if len(valleys) > 0: return first_peak_idx + valleys[0]
    return first_peak_idx

def extractfeatures(trajectories):
    feature_list = []
    
    feature_names = [
        'X_iqr', 'Y_iqr', 'Z_iqr',           # 0~2
        'length', 'max_reach', 'efficiency', # 3~5
        'range_X', 'range_Y', 'range_Z',     # 6~8
        'ratio_pca',                         # 9
        'jerk_smooth',                       # 10
        'xy_area',                           # 11 
        'slope_xy',                          # 12 
        'corr_xy',                           # 13 
        'apex_vec_x', 'apex_vec_y',          # 14, 15
        'apex_vec_z',                        # 16 
        'radius_ratio',                      # 17
        'pca_z',                             # 18 
        'line_fit_rmse',                     # 19
        'deviation_max',                     # 20 
        'turn_angle',                        # 21
        'ldlj',                              # 22
        'dwt_energy_detail'                  # 23
    ]

    for traj_raw in trajectories:
        if len(traj_raw) < 5:
            feature_list.append(np.zeros(len(feature_names)))
            continue

        # [핵심 수정: Dual-Stream Smoothing Strategy]
        # 1. traj_shape: 기존 Baseline보다 더 강력한 스무딩 (sigma=3.0)
        #    -> 목적: Drift(곡선)를 직선으로 펴서 형상 인식 정확도 극대화
        traj_shape = gaussian_filter1d(traj_raw, sigma=3.0, axis=0)
        
        # 2. traj_detail: 약한 스무딩 (sigma=1.0)
        #    -> 목적: 떨림, Jerk, LDLJ 등 세밀한 움직임 품질 분석
        traj_detail = gaussian_filter1d(traj_raw, sigma=1.0, axis=0)

        start_point = traj_shape[0] # 기준점은 Shape 궤적 사용

        # ---------------------------------------------------
        # [데이터 분리] - 기준은 traj_shape (Robust)
        # ---------------------------------------------------
        apex_point_global, _ = get_robust_apex(traj_shape, start_point)
        max_reach = np.linalg.norm(apex_point_global - start_point)

        end_idx_oneway = get_one_way_end_idx(traj_shape, start_point)
        traj_oneway = traj_shape[:end_idx_oneway+1] # One-way도 Shape 기준
        
        start_oneway = traj_oneway[0]
        dists_ow = np.linalg.norm(traj_oneway - start_oneway, axis=1)
        max_reach_oneway = np.max(dists_ow)
        
        diffs_oneway = np.diff(traj_oneway, axis=0)
        length_oneway = np.sum(np.linalg.norm(diffs_oneway, axis=1))

        if length_oneway < max_reach * 0.2:
            traj_oneway = traj_shape 
            length_oneway = np.sum(np.linalg.norm(np.diff(traj_shape, axis=0), axis=1))
            max_reach_oneway = max_reach

        # ---------------------------------------------------
        # 피쳐 계산 (대부분 traj_shape 사용 -> Drift 극복)
        # ---------------------------------------------------

        # [Global] 0~2. IQR
        iqrs = iqr(traj_shape, axis=0, rng=(25, 75))

        # [Global] 3. Length
        diffs = np.diff(traj_shape, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        
        # [One-way] 5. Efficiency
        efficiency = 0.0
        if length_oneway > 1e-3:
            apex_idx_local = np.argmax(dists_ow)
            len_to_apex = np.sum(np.linalg.norm(np.diff(traj_oneway[:apex_idx_local+1], axis=0), axis=1))
            if len_to_apex > 1e-3:
                efficiency = max_reach_oneway / len_to_apex 

        # [Global] 6~8. Range
        p_min = np.percentile(traj_shape, 2, axis=0)
        p_max = np.percentile(traj_shape, 98, axis=0)
        bounding_box = p_max - p_min

        # [Global] 9, 18. PCA (중요: 휘어진 직선을 PCA하면 Ratio가 나빠지는데, 강한 스무딩이 이를 보정함)
        pca = PCA(n_components=2)
        pca.fit(traj_shape)
        variances = pca.explained_variance_
        ratio_pca = variances[1] / variances[0] if variances[0] > 1e-6 else 0.0
        pca_z = abs(pca.components_[0][2])

        # [Global] 10. Jerk (Simple)
        acc = np.diff(diffs, axis=0)
        jerk_vec = np.diff(acc, axis=0)
        total_jerk = np.sum(np.linalg.norm(jerk_vec, axis=1))
        if length > 1e-3: total_jerk /= length

        # [One-way] 11. XY Area
        xy_area = 0.0
        try:
            if len(traj_oneway) > 3:
                hull = ConvexHull(traj_oneway[:, :2])
                xy_area = hull.volume
        except: pass

        # [One-way] 12, 13. Slope, Correlation
        slope_xy, corr_xy = 0.0, 0.0
        try:
            if len(traj_oneway) > 2:
                lr = LinearRegression()
                lr.fit(traj_oneway[:, 0].reshape(-1, 1), traj_oneway[:, 1])
                slope_xy = lr.coef_[0]
                c = np.corrcoef(traj_oneway[:, 0], traj_oneway[:, 1])[0, 1]
                if not np.isnan(c): corr_xy = c
        except: pass

        # [Global] 14~16. Apex Vector
        apex_vec_x, apex_vec_y, apex_vec_z = 0.0, 0.0, 0.0
        try:
            if max_reach > 1e-6:
                apex_vec_x = (apex_point_global[0] - start_point[0]) / max_reach
                apex_vec_y = (apex_point_global[1] - start_point[1]) / max_reach
                apex_vec_z = (apex_point_global[2] - start_point[2]) / max_reach
        except: pass

        # [One-way] 17. Radius Ratio
        radius_ratio = 0.0
        try:
            if len(traj_oneway) > 2:
                centroid = np.mean(traj_oneway, axis=0)
                dists_center = np.linalg.norm(traj_oneway - centroid, axis=1)
                r_min = np.percentile(dists_center, 5)
                r_max = np.percentile(dists_center, 95)
                if r_max > 1e-6: radius_ratio = r_min / r_max
        except: pass

        # [One-way] 19. Line Fit RMSE (Drift에 가장 취약한 피쳐 -> 강한 스무딩으로 해결)
        line_fit_rmse = 0.0
        try:
            if len(traj_oneway) > 2:
                pca_ow = PCA(n_components=2)
                pca_ow.fit(traj_oneway)
                var_ow = pca_ow.explained_variance_
                if var_ow[0] > 1e-6:
                    line_fit_rmse = np.sqrt(var_ow[1]) / (max_reach_oneway + 1e-6)
        except: pass

        # [One-way] 20. Deviation Max
        deviation_max = 0.0
        if max_reach_oneway > 1e-3:
            apex_vec_ow = traj_oneway[np.argmax(dists_ow)] - start_oneway
            vec_to_points = traj_oneway - start_oneway
            cross_prods = np.cross(vec_to_points, apex_vec_ow)
            distances = np.linalg.norm(cross_prods, axis=1) / max_reach_oneway
            deviation_max = np.max(distances)

        # [One-way] 21. Turn Angle
        turn_angle = 0.0
        window = 5 
        apex_idx_ow = np.argmax(dists_ow)
        p_idx = apex_idx_ow
        prev_idx = max(0, p_idx - window)
        next_idx = min(len(traj_oneway) - 1, p_idx + window)
        if next_idx > prev_idx:
            vec_in = traj_oneway[p_idx] - traj_oneway[prev_idx]
            vec_out = traj_oneway[next_idx] - traj_oneway[p_idx]
            n_in = np.linalg.norm(vec_in)
            n_out = np.linalg.norm(vec_out)
            if n_in > 1e-6 and n_out > 1e-6:
                turn_angle = np.dot(vec_in, vec_out) / (n_in * n_out)

        # ---------------------------------------------------
        # [신규 추가] 22, 23 피쳐 (traj_detail 사용)
        # ---------------------------------------------------
        
        # [22] LDLJ (Log Dimensionless Jerk)
        ldlj = 0.0
        try:
            # 여기서는 너무 강한 스무딩(traj_shape)을 쓰면 안됨.
            # 적당히 노이즈만 잡은 traj_detail 사용
            vel = np.diff(traj_detail, axis=0)
            acc = np.diff(vel, axis=0)
            jerk = np.diff(acc, axis=0)
            
            jerk_sq_sum = np.sum(np.linalg.norm(jerk, axis=1)**2)
            max_vel = np.max(np.linalg.norm(vel, axis=1)) + 1e-6
            duration = len(traj_detail)
            
            if jerk_sq_sum > 1e-9:
                dim_jerk = (jerk_sq_sum * (duration**3)) / (max_vel**2)
                ldlj = -np.log(dim_jerk + 1e-9) 
        except: pass

        # [23] DWT Energy (High Frequency Wobble)
        dwt_energy = 0.0
        try:
            # DWT도 디테일이 살아있는 데이터 사용
            coeffs = pywt.wavedec(traj_detail, 'db4', level=2, axis=0)
            detail_coeffs = coeffs[1:] 
            energy_sum = sum([np.sum(c**2) for c in detail_coeffs])
            dwt_energy = np.log(energy_sum + 1e-6) 
        except: pass

        # 최종 병합
        features = np.concatenate([
            iqrs,                            # 0~2
            [length, max_reach, efficiency], # 3~5
            bounding_box,                    # 6~8
            [ratio_pca],                     # 9
            [total_jerk],                    # 10
            [xy_area],                       # 11
            [slope_xy],                      # 12
            [corr_xy],                       # 13
            [apex_vec_x, apex_vec_y, apex_vec_z], # 14~16
            [radius_ratio],                  # 17
            [pca_z],                         # 18
            [line_fit_rmse],                 # 19
            [deviation_max],                 # 20
            [turn_angle],                    # 21
            [ldlj, dwt_energy]               # 22, 23
        ])
        feature_list.append(features)

    return np.array(feature_list), feature_names