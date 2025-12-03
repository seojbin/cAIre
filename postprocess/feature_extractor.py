import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from scipy.stats import iqr
from scipy.ndimage import gaussian_filter1d

def get_robust_apex(traj, start_point):
    dists = np.linalg.norm(traj - start_point, axis=1)
    if len(traj) < 10:
        return traj[np.argmax(dists)], np.argmax(dists)

    threshold = np.percentile(dists, 85)
    candidate_indices = np.where(dists >= threshold)[0]
    candidates = traj[candidate_indices]

    if len(candidates) == 0:
        return traj[np.argmax(dists)], np.argmax(dists)

    geo_median = np.median(candidates, axis=0)
    dists_from_median = np.linalg.norm(candidates - geo_median, axis=1)
    limit_dist = np.median(dists_from_median) * 2.0 + 1e-6
    mask = dists_from_median <= limit_dist
    
    clean_candidates = candidates[mask]
    if len(clean_candidates) == 0: clean_candidates = candidates

    robust_apex = np.mean(clean_candidates, axis=0)
    dists_to_robust = np.linalg.norm(traj - robust_apex, axis=1)
    robust_idx = np.argmin(dists_to_robust)
    
    return robust_apex, robust_idx

def extractfeatures(trajectories):
    feature_list = []

    feature_names = [
        'X_iqr', 'Y_iqr', 'Z_iqr',           # 0, 1, 2
        'length', 'max_reach', 'efficiency', # 3, 4, 5 [Updated]
        'range_X', 'range_Y', 'range_Z',     # 6, 7, 8
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
        'turn_angle'                         # 21
    ]

    for traj in trajectories:
        if len(traj) < 5:
            feature_list.append(np.zeros(len(feature_names)))
            continue

        # 1. 스무딩
        traj_smooth = gaussian_filter1d(traj, sigma=2.0, axis=0)
        start_point = traj_smooth[0]

        apex_point, apex_idx = get_robust_apex(traj_smooth, start_point)
        max_reach = np.linalg.norm(apex_point - start_point)

        # ----------------------------------------------------------
        # 피쳐 계산
        # ----------------------------------------------------------
        
        # 0~2. IQR
        iqrs = iqr(traj, axis=0, rng=(25, 75))

        # 3. Length (Full resolution)
        diffs = np.diff(traj_smooth, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        
        # [UPDATED 5] Robust Efficiency (Downsampled)
        # 가우시안 노이즈(자글자글함)를 무시하기 위해 5스텝마다 샘플링하여 길이 계산
        # 이러면 떨림이 있어도 전체 이동 거리가 뻥튀기되지 않음
        step = 5
        if len(traj_smooth) > step:
            downsampled = traj_smooth[::step]
            diffs_down = np.diff(downsampled, axis=0)
            len_down = np.sum(np.linalg.norm(diffs_down, axis=1))
        else:
            len_down = length

        efficiency = 1.0
        if max_reach > 1e-3:
            # 다운샘플링된 길이로 효율성 계산 (노이즈에 훨씬 강함)
            efficiency = len_down / (2 * max_reach)

        # 6~8. Range
        p_min = np.percentile(traj, 2, axis=0)
        p_max = np.percentile(traj, 98, axis=0)
        bounding_box = p_max - p_min

        # 9, 18. PCA
        pca = PCA(n_components=2)
        pca.fit(traj_smooth)
        variances = pca.explained_variance_
        ratio_pca = variances[1] / variances[0] if variances[0] > 1e-6 else 0.0
        pca_z = abs(pca.components_[0][2])

        # [REPLACED 19] Line Fit RMSE (직선과의 오차)
        # origin_dist_cv 대신, 데이터를 직선에 피팅했을 때의 잔차(Residual)를 사용
        # 직선 운동: 잔차가 매우 작음 (노이즈가 있어도 평균 중심선 근처임)
        # 원 운동: 잔차가 매우 큼 (직선으로 설명 불가능)
        # PCA의 두 번째 분산값이 사실상 '주축에서 벗어난 거리의 제곱 평균'과 유사함
        line_fit_rmse = 0.0
        if variances[0] > 1e-6:
            # 정규화: (단축 분산 / 장축 분산)의 제곱근 -> 직선에서 퍼진 정도
            # ratio_pca와 비슷하지만, 길이 정규화를 위해 max_reach로 나눈 효과
            line_fit_rmse = np.sqrt(variances[1]) / (max_reach + 1e-6)

        # 10. Jerk
        acc = np.diff(diffs, axis=0)
        jerk_vec = np.diff(acc, axis=0)
        total_jerk = np.sum(np.linalg.norm(jerk_vec, axis=1))
        if length > 1e-3: total_jerk /= length

        # 11. XY Area 
        xy_area = 0.0
        try:
            hull = ConvexHull(traj_smooth[:, :2])
            xy_area = hull.volume
        except: pass

        # 12. Slope 
        slope_xy = 0.0
        try:
            lr = LinearRegression()
            lr.fit(traj_smooth[:, 0].reshape(-1, 1), traj_smooth[:, 1])
            slope_xy = lr.coef_[0]
        except: pass

        # 13. Correlation 
        corr_xy = 0.0
        try:
            c = np.corrcoef(traj_smooth[:, 0], traj_smooth[:, 1])[0, 1]
            if not np.isnan(c): corr_xy = c
        except: pass

        # 14, 15, 16. Apex Vector
        apex_vec_x, apex_vec_y, apex_vec_z = 0.0, 0.0, 0.0
        try:
            mag = np.linalg.norm(apex_point - start_point)
            if mag > 1e-6:
                apex_vec_x = (apex_point[0] - start_point[0]) / mag
                apex_vec_y = (apex_point[1] - start_point[1]) / mag
                apex_vec_z = (apex_point[2] - start_point[2]) / mag
        except: pass

        # 17. Radius Ratio
        radius_ratio = 0.0
        try:
            centroid = np.mean(traj_smooth, axis=0)
            dists_center = np.linalg.norm(traj_smooth - centroid, axis=1)
            r_min = np.percentile(dists_center, 5)
            r_max = np.percentile(dists_center, 95)
            if r_max > 1e-6: radius_ratio = r_min / r_max
        except: pass

        # 20. Deviation Max
        deviation_max = 0.0
        if max_reach > 1e-3:
            apex_vec = apex_point - start_point
            vec_to_points = traj_smooth - start_point
            cross_prods = np.cross(vec_to_points, apex_vec)
            distances = np.linalg.norm(cross_prods, axis=1) / max_reach
            deviation_max = np.max(distances)

        # 21. Turn Angle
        turn_angle = 0.0
        window = 5 
        prev_idx = max(0, apex_idx - window)
        next_idx = min(len(traj_smooth) - 1, apex_idx + window)
        
        if next_idx > prev_idx:
            vec_in = traj_smooth[apex_idx] - traj_smooth[prev_idx]
            vec_out = traj_smooth[next_idx] - traj_smooth[apex_idx]
            norm_in = np.linalg.norm(vec_in)
            norm_out = np.linalg.norm(vec_out)
            if norm_in > 1e-6 and norm_out > 1e-6:
                turn_angle = np.dot(vec_in, vec_out) / (norm_in * norm_out)

        # 최종 병합
        features = np.concatenate([
            iqrs,                            # 0, 1, 2
            [length, max_reach, efficiency], # 3, 4, 5 [Robust]
            bounding_box,                    # 6, 7, 8
            [ratio_pca],                     # 9
            [total_jerk],                    # 10
            [xy_area],                       # 11 
            [slope_xy],                      # 12 
            [corr_xy],                       # 13 
            [apex_vec_x, apex_vec_y, apex_vec_z], # 14, 15, 16
            [radius_ratio],                  # 17
            [pca_z],                         # 18
            [line_fit_rmse],                 # 19 [REPLACED]
            [deviation_max],                 # 20 
            [turn_angle]                     # 21
        ])
        feature_list.append(features)

    return np.array(feature_list), feature_names