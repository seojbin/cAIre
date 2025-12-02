#3D 궤적 데이터 리스트를 2D 배열로 특성 추출해 변환하는 코드
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull, QhullError


def extractfeatures(trajectories):
    feature_list = []
    # 피쳐 이름 (총 16개로 변경: 14, 15는 Apex Vector)
    feature_names = [
        'X_std', 'Y_std', 'Z_std',  # 0, 1, 2
        'length', 'disp', 'length_disp',  # 3, 4, 5
        'range_X', 'range_Y', 'range_Z',  # 6, 7, 8
        'ratio',  # 9
        'jerk',  # 10
        'xy_area',  # 11
        'slope_xy',  # 12
        'corr_xy',  # 13
        'apex_vec_x', 'apex_vec_y'  # 14, 15
    ]

    for traj in trajectories:
        stds = np.std(traj, axis=0)
        mins = np.min(traj, axis=0)
        maxs = np.max(traj, axis=0)

        diffs = np.diff(traj, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        start = traj[0]
        end = traj[-1]
        disp = np.linalg.norm(end - start)
        length_disp = length / disp if disp > 1e-6 else length

        bounding_box = maxs - mins

        corr_xy = 0.0
        try:
            c = np.corrcoef(traj[:, 0], traj[:, 1])[0, 1]
            if not np.isnan(c): corr_xy = c
        except:
            pass

        ratio = 0.0
        total_jerk = 0.0
        xy_area = 0.0
        slope_xy = 0.0

        # Apex Vector 계산
        apex_vec_x = 0.0
        apex_vec_y = 0.0
        if len(traj) > 1:
            dist_from_start = np.linalg.norm(traj - start, axis=1)
            apex_idx = np.argmax(dist_from_start)
            apex_point = traj[apex_idx]
            apex_vec = apex_point - start
            magnitude = np.linalg.norm(apex_vec)
            if magnitude > 1e-6:
                apex_vec_x = apex_vec[0] / magnitude  # X 성분 정규화
                apex_vec_y = apex_vec[1] / magnitude  # Y 성분 정규화
            else:
                apex_vec_x = 0.0
                apex_vec_y = 0.0
            apex_vec_x = apex_vec[0]
            apex_vec_y = apex_vec[1]

        if len(traj) > 4:
            try:
                pca = PCA(n_components=1)
                pca.fit(traj)
                ratio = pca.explained_variance_ratio_[0]

                acc = np.diff(diffs, axis=0)
                jerk = np.diff(acc, axis=0)
                total_jerk = np.sum(np.linalg.norm(jerk, axis=1))

                hull = ConvexHull(traj[:, :2])
                xy_area = hull.volume

                lr = LinearRegression()
                lr.fit(traj[:, 0].reshape(-1, 1), traj[:, 1])
                slope_xy = lr.coef_[0]
            except:
                pass

        features = np.concatenate([
            stds, [length, disp, length_disp], bounding_box,
            [ratio], [total_jerk], [xy_area], [slope_xy], [corr_xy],
            [apex_vec_x, apex_vec_y]
        ])
        feature_list.append(features)

    return np.array(feature_list), feature_names