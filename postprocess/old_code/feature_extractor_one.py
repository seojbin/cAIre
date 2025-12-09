import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from scipy.stats import iqr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def get_robust_apex(traj, start_point):
    """ [Global Apex] Median 기반, 스파이크 노이즈에 강인함 """
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
    """ [One-way Finder] 강력한 스무딩으로 노이즈 무시하고 반환점 찾기 """
    dists = np.linalg.norm(traj - start_point, axis=1)
    
    # Peak 탐색용 강력한 스무딩 (sigma=5.0)
    dists_smooth = gaussian_filter1d(dists, sigma=5.0)
    
    global_max = np.max(dists_smooth)
    min_reach_threshold = global_max * 0.5 
    
    peaks, _ = find_peaks(dists_smooth, height=min_reach_threshold, distance=10)
    
    if len(peaks) == 0:
        return np.argmax(dists)
        
    first_peak_idx = peaks[0]
    
    # Peak 이후의 골짜기 탐색
    post_peak_dists = dists_smooth[first_peak_idx:]
    if len(post_peak_dists) < 5:
        return first_peak_idx
        
    valleys, _ = find_peaks(-post_peak_dists, distance=10)
    
    if len(valleys) > 0:
        return first_peak_idx + valleys[0]
    
    return len(traj) - 1 

def extractfeatures(trajectories):
    feature_list = []
    
    feature_names = [
        'X_iqr', 'Y_iqr', 'Z_iqr',           # 0~2 (Global)
        'length', 'max_reach', 'efficiency', # 3, 4, 5 (Robust Logic)
        'range_X', 'range_Y', 'range_Z',     # 6~8 (Global)
        'ratio_pca',                         # 9 (Global)
        'jerk_smooth',                       # 10 (Structural)
        'xy_area',                           # 11 (One-way/Global)
        'slope_xy',                          # 12 (One-way/Global)
        'corr_xy',                           # 13 (One-way/Global)
        'apex_vec_x', 'apex_vec_y',          # 14, 15 (Global)
        'apex_vec_z',                        # 16 (Global)
        'radius_ratio',                      # 17 (One-way/Global)
        'pca_z',                             # 18 (Global)
        'line_fit_rmse',                     # 19 (One-way/Global)
        'deviation_max',                     # 20 (One-way/Global)
        'turn_angle'                         # 21 (One-way/Global)
    ]

    for traj in trajectories:
        if len(traj) < 5:
            feature_list.append(np.zeros(len(feature_names)))
            continue

        # 1. 기본 스무딩 (형상 파악용)
        traj_smooth = gaussian_filter1d(traj, sigma=2.0, axis=0)
        start_point = traj_smooth[0]

        # 2. 구조적 스무딩 (Efficiency/Jerk/Length용 - 노이즈 제거)
        traj_structural = gaussian_filter1d(traj, sigma=5.0, axis=0)

        # ---------------------------------------------------
        # [A. Global Features]
        # ---------------------------------------------------
        apex_point_global, _ = get_robust_apex(traj_smooth, start_point)
        max_reach_global = np.linalg.norm(apex_point_global - start_point)

        # ---------------------------------------------------
        # [B. One-way Segment Detection & SAFETY LOCK]
        # ---------------------------------------------------
        end_idx = get_one_way_end_idx(traj_smooth, start_point)
        
        # [검증] 잘라낸 구간의 길이(변위) 확인
        # 노이즈로 인해 시작하자마자 잘렸는지 확인 (Safety Lock)
        dists_check = np.linalg.norm(traj_smooth[:end_idx+1] - start_point, axis=1)
        local_reach = np.max(dists_check) if len(dists_check) > 0 else 0
        
        if local_reach < max_reach_global * 0.2:
            traj_target = traj_smooth         # 전체 사용
            traj_target_struct = traj_structural # 전체 사용
        else:
            traj_target = traj_smooth[:end_idx+1]          # One-way 사용
            traj_target_struct = traj_structural[:end_idx+1] # One-way 사용

        # ---------------------------------------------------
        # 피쳐 계산 (traj_target: One-way 혹은 Global Fallback)
        # ---------------------------------------------------

        # [Global] 0~2. IQR
        iqrs = iqr(traj, axis=0, rng=(25, 75))

        # [Target] 3. Length (Structural - 해안선 역설 방지)
        diffs_struct = np.diff(traj_target_struct, axis=0)
        length_struct = np.sum(np.linalg.norm(diffs_struct, axis=1))
        length = length_struct
        
        # [Global] 4. Max Reach
        max_reach = max_reach_global
        
        # [Target] 5. Efficiency (Structural Length 사용)
        dists_target = np.linalg.norm(traj_target - start_point, axis=1)
        max_reach_target = np.max(dists_target)
        
        # 효율성: (Apex까지의 변위) / (Apex까지의 실제 이동거리)
        # 분모는 반드시 Structural(매끄러운) 궤적을 사용해야 노이즈에 안 무너짐
        efficiency = 0.0
        apex_idx_local = np.argmax(dists_target)
        
        # Apex까지의 경로 길이 (Structural 기준)
        path_to_apex_struct = traj_target_struct[:apex_idx_local+1]
        len_to_apex_struct = np.sum(np.linalg.norm(np.diff(path_to_apex_struct, axis=0), axis=1))
        
        if len_to_apex_struct > 1e-3:
            efficiency = max_reach_target / len_to_apex_struct

        # [Global] 6~8. Range
        p_min = np.percentile(traj, 2, axis=0)
        p_max = np.percentile(traj, 98, axis=0)
        bounding_box = p_max - p_min

        # [Global] 9, 18. PCA
        pca = PCA(n_components=2)
        pca.fit(traj_smooth)
        variances = pca.explained_variance_
        ratio_pca = variances[1] / variances[0] if variances[0] > 1e-6 else 0.0
        pca_z = abs(pca.components_[0][2])

        # [Target] 10. Jerk (Structural - 미분 폭주 방지)
        acc = np.diff(diffs_struct, axis=0)
        jerk_vec = np.diff(acc, axis=0)
        total_jerk = np.sum(np.linalg.norm(jerk_vec, axis=1))
        if length > 1e-3: total_jerk /= length

        # [Target] 11. XY Area
        xy_area = 0.0
        try:
            if len(traj_target) > 3:
                hull = ConvexHull(traj_target[:, :2])
                xy_area = hull.volume
        except: pass

        # [Target] 12, 13. Slope, Corr
        slope_xy, corr_xy = 0.0, 0.0
        try:
            if len(traj_target) > 2:
                lr = LinearRegression()
                lr.fit(traj_target[:, 0].reshape(-1, 1), traj_target[:, 1])
                slope_xy = lr.coef_[0]
                c = np.corrcoef(traj_target[:, 0], traj_target[:, 1])[0, 1]
                if not np.isnan(c): corr_xy = c
        except: pass

        # [Global] 14~16. Apex Vector
        apex_vec = np.zeros(3)
        if max_reach > 1e-6:
            apex_vec = (apex_point_global - start_point) / max_reach

        # [Target] 17. Radius Ratio
        radius_ratio = 0.0
        try:
            centroid = np.mean(traj_target, axis=0)
            dists_center = np.linalg.norm(traj_target - centroid, axis=1)
            r_min = np.percentile(dists_center, 5)
            r_max = np.percentile(dists_center, 95)
            if r_max > 1e-6: radius_ratio = r_min / r_max
        except: pass

        # [Target] 19. Line Fit RMSE
        line_fit_rmse = 0.0
        try:
            if len(traj_target) > 2:
                pca_ow = PCA(n_components=2)
                pca_ow.fit(traj_target)
                var_ow = pca_ow.explained_variance_
                if var_ow[0] > 1e-6:
                    line_fit_rmse = np.sqrt(var_ow[1]) / (max_reach_target + 1e-6)
        except: pass

        # [Target] 20. Deviation Max
        deviation_max = 0.0
        if max_reach_target > 1e-3:
            apex_vec_ow = traj_target[np.argmax(dists_target)] - start_point
            vec_to_points = traj_target - start_point
            cross_prods = np.cross(vec_to_points, apex_vec_ow)
            distances = np.linalg.norm(cross_prods, axis=1) / max_reach_target
            deviation_max = np.max(distances)

        # [Target] 21. Turn Angle
        turn_angle = 0.0
        window = 5 
        apex_idx_ret = np.argmax(dists_target)
        p_idx = apex_idx_ret
        prev_idx = max(0, p_idx - window)
        next_idx = min(len(traj_target) - 1, p_idx + window)
        
        if next_idx > prev_idx:
            vec_in = traj_target[p_idx] - traj_target[prev_idx]
            vec_out = traj_target[next_idx] - traj_target[p_idx]
            n_in = np.linalg.norm(vec_in)
            n_out = np.linalg.norm(vec_out)
            if n_in > 1e-6 and n_out > 1e-6:
                turn_angle = np.dot(vec_in, vec_out) / (n_in * n_out)

        # 최종 병합
        features = np.concatenate([
            iqrs,                        # 0~2
            [length, max_reach, efficiency], # 3~5
            bounding_box,                # 6~8
            [ratio_pca],                 # 9
            [total_jerk],                # 10
            [xy_area],                   # 11
            [slope_xy],                  # 12
            [corr_xy],                   # 13
            apex_vec,                    # 14~16
            [radius_ratio],              # 17
            [pca_z],                     # 18
            [line_fit_rmse],             # 19
            [deviation_max],             # 20
            [turn_angle]                 # 21
        ])
        feature_list.append(features)

    return np.array(feature_list), feature_names