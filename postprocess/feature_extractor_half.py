import numpy as np

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

from scipy.spatial import ConvexHull

from scipy.stats import iqr

from scipy.ndimage import gaussian_filter1d

from scipy.signal import find_peaks



def get_robust_apex(traj, start_point):

    """ [Global Apex] 전체 궤적 중 가장 멀리 있는 점들의 중심 (위치/방향용) """

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

    # 원본 궤적 매칭

    robust_idx = np.argmin(np.linalg.norm(traj - robust_apex, axis=1))

    

    return robust_apex, robust_idx



def get_one_way_end_idx(traj, start_point):

    """

    [One-way End Finder]

    전략: 시작점에서 멀어졌다가(Peak), 다시 돌아와서 가장 가까워진 지점(Valley)을 찾는다.

    만약 돌아오지 않는다면(직선 뻗기), 최대 거리 지점(Apex)을 반환한다.

    """

    dists = np.linalg.norm(traj - start_point, axis=1)

    

    # 1. 스무딩 (노이즈 제거)

    dists_smooth = gaussian_filter1d(dists, sigma=3.0)

    

    # 2. Apex(최대 거리) 찾기 (적어도 여기까지는 가야 함)

    #    너무 초반에 돌아오는 건 노이즈임. 전체 Max의 50% 이상은 갔다가 와야 인정.

    global_max_dist = np.max(dists_smooth)

    min_reach_threshold = global_max_dist * 0.5

    

    # Peak 탐색 (높이가 50% 이상인 것 중 첫 번째)

    peaks, _ = find_peaks(dists_smooth, height=min_reach_threshold, distance=10)

    

    if len(peaks) == 0:

        # 멀어지기만 하고 굴곡이 없으면 그냥 전체 Max 지점 반환

        return np.argmax(dists)

        

    first_peak_idx = peaks[0]

    

    # 3. Peak 이후에 "돌아오는 지점(Local Minima)" 탐색

    #    Peak 이후의 데이터만 잘라서 봅니다.

    post_peak_dists = dists_smooth[first_peak_idx:]

    

    # valley 찾기 (뒤집어서 find_peaks 쓰면 됨)

    valleys, _ = find_peaks(-post_peak_dists, distance=10)

    

    if len(valleys) > 0:

        # Peak 이후 첫 번째 골짜기 = 1회 왕복 완료 지점

        return first_peak_idx + valleys[0]

    

    # 돌아오는 지점을 못 찾았으면 (갔다가 멈췄거나, 끝까지 안 돌아옴)

    # 그냥 Peak 지점(Apex)까지만 자름 -> 이러면 '가는 길'만 남음 (Half-cycle)

    return first_peak_idx



def extractfeatures(trajectories):

    feature_list = []

    

    # [피쳐 목록 유지] 삭제 없이 모든 피쳐 인덱스 보존

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

        'turn_angle'                         # 21

    ]



    for traj in trajectories:

        if len(traj) < 5:

            feature_list.append(np.zeros(len(feature_names)))

            continue



        # 1. 기본 스무딩

        traj_smooth = gaussian_filter1d(traj, sigma=2.0, axis=0)

        start_point = traj_smooth[0]



        # ---------------------------------------------------

        # [데이터 분리] 

        # A. Global: 전체 궤적 (위치, 범위, PCA Ratio용)

        # B. One-way: 1회 운동 구간 (형상, 효율성, Area용)

        # ---------------------------------------------------

        

        # A. Global Apex

        apex_point_global, _ = get_robust_apex(traj_smooth, start_point)

        max_reach = np.linalg.norm(apex_point_global - start_point)



        # B. One-way Slice

        # 사용자 요청: "돌아오는 반환점을 인정하는 방식" 적용

        end_idx_oneway = get_one_way_end_idx(traj_smooth, start_point)

        traj_oneway = traj_smooth[:end_idx_oneway+1]

        

        # One-way 기준 값

        start_oneway = traj_oneway[0]

        end_oneway = traj_oneway[-1]

        # One-way 내에서의 Apex (최대 도달점) 재계산 (Efficiency 분자용)

        dists_ow = np.linalg.norm(traj_oneway - start_oneway, axis=1)

        max_reach_oneway = np.max(dists_ow)

        

        diffs_oneway = np.diff(traj_oneway, axis=0)

        length_oneway = np.sum(np.linalg.norm(diffs_oneway, axis=1))



        # [Safety Lock] 너무 짧으면 Global로 대체

        if length_oneway < max_reach * 0.2:

            traj_oneway = traj_smooth 

            length_oneway = np.sum(np.linalg.norm(np.diff(traj_smooth, axis=0), axis=1))

            max_reach_oneway = max_reach



        # ---------------------------------------------------

        # 피쳐 계산 (Global vs One-way 적절히 배분)

        # ---------------------------------------------------



        # [Global] 0~2. IQR

        iqrs = iqr(traj, axis=0, rng=(25, 75))



        # [Global] 3. Length (전체 운동량)

        diffs = np.diff(traj_smooth, axis=0)

        length = np.sum(np.linalg.norm(diffs, axis=1))

        

        # [One-way] 5. Efficiency (핵심)

        # 돌아오는 지점까지 포함했으므로, 분자는 '변위'가 아니라 '최대 도달 거리(Apex) * 2'가 되어야 함?

        # 아니면 기존처럼 'Apex까지 거리 / 이동거리'로 할까?

        # -> 사용자 요청("반환점 인정")에 따르면, "갔다 온 거리" 대비 "Apex 거리" 비율이 적절함.

        # 직선 왕복: 2 * Reach / Length ≈ 1.0

        # 원 왕복: 2 * Reach / Length ≈ 2*R / (pi*R) = 0.63

        efficiency = 0.0

        if length_oneway > 1e-3:

            # 왕복 운동으로 가정하고 계산 (Apex * 2 / Length)

            # 만약 편도(갔다 멈춤)라면 Apex / Length ≈ 1.0이 되므로 호환됨.

            # *주의: 여기서 2를 곱하면 편도일 때 2.0이 되버림. 

            # 가장 안전한 건 "Apex까지의 거리 / (해당 시점까지의 길이)"로 계산하는 Half-Efficiency임.

            # 하지만 요청하신대로 '반환점'까지 끊었다면, 아래 공식이 범용적임:

            efficiency = (2 * max_reach_oneway) / length_oneway 

            # (직선 왕복 시 1.0, 원 왕복 시 0.63, 직선 편도 시 2.0 -> 편도 직선은 튀게 됨. 감안해야 함)

            

            # [수정 제안] 편도/왕복 혼재 시 가장 강인한 건 여전히 "Apex까지의 Half Efficiency"입니다.

            # 그래도 요청하신 로직(Trajectory 전체 사용)을 최대한 살리기 위해

            # One-way Trajectory 내부에서 다시 Apex까지의 길이를 구해서 쓰는게 안전합니다.

            apex_idx_local = np.argmax(dists_ow)

            len_to_apex = np.sum(np.linalg.norm(np.diff(traj_oneway[:apex_idx_local+1], axis=0), axis=1))

            if len_to_apex > 1e-3:

                efficiency = max_reach_oneway / len_to_apex # 이게 제일 안전 (Always 0~1)



        # [Global] 6~8. Range

        p_min = np.percentile(traj, 2, axis=0)

        p_max = np.percentile(traj, 98, axis=0)

        bounding_box = p_max - p_min



        # [Global] 9, 18. PCA (전체 형상의 뚱뚱함 파악 - False Circle 방지용)

        pca = PCA(n_components=2)

        pca.fit(traj_smooth) # 전체 사용

        variances = pca.explained_variance_

        ratio_pca = variances[1] / variances[0] if variances[0] > 1e-6 else 0.0

        pca_z = abs(pca.components_[0][2])



        # [Global] 10. Jerk

        acc = np.diff(diffs, axis=0)

        jerk_vec = np.diff(acc, axis=0)

        total_jerk = np.sum(np.linalg.norm(jerk_vec, axis=1))

        if length > 1e-3: total_jerk /= length



        # [One-way] 11. XY Area (왕복 면적 노이즈 제거)

        xy_area = 0.0

        try:

            if len(traj_oneway) > 3:

                hull = ConvexHull(traj_oneway[:, :2])

                xy_area = hull.volume

        except: pass



        # [One-way] 12, 13. Slope, Correlation (복구됨)

        slope_xy, corr_xy = 0.0, 0.0

        try:

            if len(traj_oneway) > 2:

                lr = LinearRegression()

                lr.fit(traj_oneway[:, 0].reshape(-1, 1), traj_oneway[:, 1])

                slope_xy = lr.coef_[0]

                

                c = np.corrcoef(traj_oneway[:, 0], traj_oneway[:, 1])[0, 1]

                if not np.isnan(c): corr_xy = c

        except: pass



        # [Global] 14~16. Apex Vector (방향성은 전체 기준)

        apex_vec_x, apex_vec_y, apex_vec_z = 0.0, 0.0, 0.0

        try:

            if max_reach > 1e-6:

                apex_vec_x = (apex_point_global[0] - start_point[0]) / max_reach

                apex_vec_y = (apex_point_global[1] - start_point[1]) / max_reach

                apex_vec_z = (apex_point_global[2] - start_point[2]) / max_reach

        except: pass



        # [One-way] 17. Radius Ratio (복구됨 - 원의 찌그러짐 판별)

        radius_ratio = 0.0

        try:

            if len(traj_oneway) > 2:

                centroid = np.mean(traj_oneway, axis=0)

                dists_center = np.linalg.norm(traj_oneway - centroid, axis=1)

                r_min = np.percentile(dists_center, 5)

                r_max = np.percentile(dists_center, 95)

                if r_max > 1e-6: radius_ratio = r_min / r_max

        except: pass



        # [One-way] 19. Line Fit RMSE

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

            # One-way Apex 사용

            apex_vec_ow = traj_oneway[np.argmax(dists_ow)] - start_oneway

            vec_to_points = traj_oneway - start_oneway

            cross_prods = np.cross(vec_to_points, apex_vec_ow)

            distances = np.linalg.norm(cross_prods, axis=1) / max_reach_oneway

            deviation_max = np.max(distances)



        # [One-way] 21. Turn Angle (Apex 기준 꺾임)

        turn_angle = 0.0

        window = 5 

        # One-way 내부에서의 Apex 인덱스 찾기 (Global 인덱스 아님)

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



        # 최종 병합 (순서 유지)

        features = np.concatenate([

            iqrs,                            # 0~2

            [length, max_reach, efficiency], # 3~5

            bounding_box,                    # 6~8

            [ratio_pca],                     # 9 (Global)

            [total_jerk],                    # 10

            [xy_area],                       # 11 (One-way)

            [slope_xy],                      # 12 (One-way)

            [corr_xy],                       # 13 (One-way)

            [apex_vec_x, apex_vec_y, apex_vec_z], # 14~16 (Global)

            [radius_ratio],                  # 17 (One-way)

            [pca_z],                         # 18

            [line_fit_rmse],                 # 19 (One-way)

            [deviation_max],                 # 20 (One-way)

            [turn_angle]                     # 21 (One-way)

        ])

        feature_list.append(features)



    return np.array(feature_list), feature_names