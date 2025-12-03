import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
from scipy.stats import iqr
from scipy.ndimage import gaussian_filter1d


def extractfeatures(trajectories):
    feature_list = []

    # [변경됨] 피쳐 이름 업데이트 (의미에 맞게 명칭 변경)
    feature_names = [
        'X_iqr', 'Y_iqr', 'Z_iqr',  # 0, 1, 2 (기존 Std 대체)
        'length', 'max_reach', 'efficiency',  # 3, 4, 5 (Disp, Ratio 대체)
        'range_X', 'range_Y', 'range_Z',  # 6, 7, 8
        'ratio_pca',  # 9
        'jerk_smooth',  # 10 (노이즈 제거된 Jerk)
        'xy_area',  # 11
        'slope_xy',  # 12
        'corr_xy',  # 13
        'apex_vec_x', 'apex_vec_y',  # 14, 15
        'radius_ratio',  # 16
        'pca_z',  # 17
        'linearity_error'  # 18
    ]

    for traj in trajectories:
        # 데이터가 너무 적으면 0으로 채움
        if len(traj) < 5:
            feature_list.append(np.zeros(len(feature_names)))
            continue

        # ==========================================
        # 1. 노이즈 제거 및 스무딩 (Robust Preprocessing)
        # ==========================================
        # 원본 데이터는 보존하되, 미분/형상 계산용 스무딩 데이터 생성
        # sigma=2.0 정도면 자잘한 떨림은 사라지고 큰 움직임만 남음
        traj_smooth = gaussian_filter1d(traj, sigma=2.0, axis=0)

        start_point = traj_smooth[0]

        # ==========================================
        # 2. 강인한 산포도 측정 (Std -> IQR)
        # ==========================================
        # Std는 튀는 값 하나에 값이 폭등함. IQR(75% - 25%)은 이상치에 강함.
        # X, Y, Z 각각의 산포도 계산
        iqrs = iqr(traj, axis=0, rng=(25, 75))

        # ==========================================
        # 3. 왕복 운동 맞춤형 거리/비율 (Robust Round-trip)
        # ==========================================
        # 길이(Length): 스무딩된 궤적으로 계산해야 노이즈로 인한 길이 뻥튀기 방지
        diffs = np.diff(traj_smooth, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))

        # Max Reach: 시작점에서 가장 멀리 떨어진 지점까지의 거리
        # (왕복 운동에서 disp는 0이 되므로, 대신 이 값을 사용)
        dists_from_start = np.linalg.norm(traj_smooth - start_point, axis=1)
        max_reach = np.max(dists_from_start) if len(dists_from_start) > 0 else 0.0

        # Efficiency: 이론상 왕복 거리(2 * max_reach) 대비 실제 이동 거리 비율
        # 1.0에 가까울수록 직선 왕복, 높을수록 복잡하게 움직임 (원, 지그재그 등)
        efficiency = 0.0
        if max_reach > 1e-3:
            efficiency = length / (2 * max_reach)
        else:
            efficiency = 1.0  # 움직임이 거의 없는 경우

        # ==========================================
        # 4. Bounding Box (Range) - 기존 유지 (Percentile 방식 우수)
        # ==========================================
        p_min = np.percentile(traj, 2, axis=0)
        p_max = np.percentile(traj, 98, axis=0)
        bounding_box = p_max - p_min

        # ==========================================
        # 5. Jerk (가속도 변화량) - 스무딩 데이터 필수
        # ==========================================
        # 노이즈가 있는 원본 데이터로 미분하면 값이 무의미해짐
        # 스무딩된 데이터로 3차 미분 수행
        acc = np.diff(diffs, axis=0)
        jerk_vec = np.diff(acc, axis=0)
        total_jerk = np.sum(np.linalg.norm(jerk_vec, axis=1))
        # 길이에 대해 정규화 (경로가 길면 Jerk 합도 커지므로)
        if length > 1e-3:
            total_jerk /= length

        # ==========================================
        # 6. 기타 형상 피쳐
        # ==========================================
        corr_xy = 0.0
        try:
            # 상관계수는 이상치에 민감하므로 스무딩 데이터 사용 권장
            c = np.corrcoef(traj_smooth[:, 0], traj_smooth[:, 1])[0, 1]
            if not np.isnan(c): corr_xy = c
        except:
            pass

        ratio_pca, xy_area, slope_xy = 0.0, 0.0, 0.0
        apex_vec_x, apex_vec_y, radius_ratio = 0.0, 0.0, 0.0
        pca_z, linearity_error = 0.0, 0.0

        try:
            # PCA Analysis
            pca = PCA(n_components=2)
            pca.fit(traj_smooth)  # 스무딩 데이터 사용
            ratio_pca = pca.explained_variance_ratio_[0]

            # Z축 변동성 (평면 벗어남 정도)
            pca_z = abs(pca.components_[0][2])
            if ratio_pca > 0: linearity_error = 1.0 - ratio_pca

            # Convex Hull (Area)
            hull = ConvexHull(traj_smooth[:, :2])
            xy_area = hull.volume

            # Robust Slope (Linear Regression)
            lr = LinearRegression()
            lr.fit(traj_smooth[:, 0].reshape(-1, 1), traj_smooth[:, 1])
            slope_xy = lr.coef_[0]

            # Apex & Radius Logic
            # Apex: 시작점에서 가장 먼 지점 10%의 평균 (Robust Max)
            sorted_indices = np.argsort(dists_from_start)[::-1]
            top_n = max(5, int(len(traj) * 0.1))
            top_points = traj_smooth[sorted_indices[:top_n]]  # 스무딩 데이터 사용

            final_apex_point = np.median(top_points, axis=0)  # Median 사용

            apex_vec = final_apex_point - start_point
            mag = np.linalg.norm(apex_vec)
            if mag > 1e-6:
                apex_vec_x = apex_vec[0] / mag
                apex_vec_y = apex_vec[1] / mag

            # Radius Ratio (Circle Check)
            # 원의 중심(평균)에서 각 점까지 거리의 균일성
            centroid = np.mean(traj_smooth, axis=0)
            dists_center = np.linalg.norm(traj_smooth - centroid, axis=1)

            # IQR 기반으로 극단적인 반지름 노이즈 제거 후 Min/Max
            r_min = np.percentile(dists_center, 5)
            r_max = np.percentile(dists_center, 95)

            if r_max > 1e-6:
                radius_ratio = r_min / r_max

        except Exception as e:
            # 에러 발생 시 0.0 유지 (print문 제거하여 로그 공해 방지)
            pass

        features = np.concatenate([
            iqrs,  # 0, 1, 2 (X, Y, Z IQR)
            [length, max_reach, efficiency],  # 3, 4, 5
            bounding_box,  # 6, 7, 8
            [ratio_pca],  # 9
            [total_jerk],  # 10
            [xy_area],  # 11
            [slope_xy],  # 12
            [corr_xy],  # 13
            [apex_vec_x, apex_vec_y],  # 14, 15
            [radius_ratio],  # 16
            [pca_z],  # 17
            [linearity_error]  # 18
        ])
        feature_list.append(features)

    return np.array(feature_list), feature_names