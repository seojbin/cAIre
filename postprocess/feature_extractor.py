#3D 궤적 데이터 리스트를 2D 배열로 특성 추출해 변환하는 코드
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression # 선형 회귀
from scipy.spatial import ConvexHull, QhullError # 2D 면적 계산

def extractfeatures(trajectories):

    
    feature_list = []
    feature_names = [
        'X_mean', 'Y_mean', 'Z_mean',
        'X_std', 'Y_std', 'Z_std',
        'X_min', 'Y_min', 'Z_min',
        'X_max', 'Y_max', 'Z_max',
        'length', 'disp', 'length_disp',
        'diff_X', 'diff_Y', 'diff_Z',
        'range_X', 'range_Y', 'range_Z',
        'ratio',
        'jerk',
        'xy_area',
        'slope_xy']

    for traj in trajectories:
        #각 축의 기본 통계
        means = np.mean(traj, axis=0)
        stds = np.std(traj, axis=0)
        mins = np.min(traj, axis=0)
        maxs = np.max(traj, axis=0)
        
        # 궤적의 전체 길이
        dist = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        length = np.sum(dist)

        # 궤적의 총 변위
        start = traj[0]
        end = traj[-1]
        disp = np.linalg.norm(end - start)
        
        # 변위 대비 길이 비율
        if disp < 1e-6:
            length_disp = length
        else:
            length_disp = length / disp

        # 시작-끝 지점의 축별 차이
        diff = end - start
        
        #바운딩 박스 크기
        bounding_box = maxs - mins
        
        #PCA 주성분 분석
        ratio = 0.0
        
        #곡률
        jerk = 0.0
        
        #2D 면적
        xy_area = 0.0

        #XY 기울기
        slope_xy = 0.0
        
        # 최소 4개 타임스텝이 있어야 Jerk/Area/Slope 계산 가능
        if len(traj) > 4:
            try:
                # PCA
                pca = PCA(n_components=1)
                pca.fit(traj)
                ratio = pca.explained_variance_ratio_[0]

                # Jerk
                # (N, 3) -> (N-1, 3) -> (N-2, 3) -> (N-3, 3)
                velocities = np.diff(traj, axis=0)
                accelerations = np.diff(velocities, axis=0)
                jerks = np.diff(accelerations, axis=0)
                total_jerk = np.sum(np.linalg.norm(jerks, axis=1))

                #2D Area
                xy_coords = traj[:, :2] # XY 평면만
                hull = ConvexHull(xy_coords)
                xy_area = hull.volume # 2D에서는 Volume이 Area
            
                #Regression Slope
                x_coords = traj[:, 0].reshape(-1, 1) # X축
                y_coords = traj[:, 1] # Y축
                lr = LinearRegression()
                lr.fit(x_coords, y_coords)
                slope_xy = lr.coef_[0] # 기울기
                
            except (QhullError, ValueError):
                # 궤적이 완벽한 직선이면 오류
                total_jerk = 0.0
                xy_area = 0.0
                slope_xy = 0.0 # 오류 시 기본값

        # 모든 특성을 하나의 리스트로 결합
        features = np.concatenate([
            means, stds, mins, maxs,
            [length, disp, length_disp],
            diff,
            bounding_box,
            [ratio],
            [total_jerk],
            [xy_area],
            [slope_xy]
        ])
        feature_list.append(features)
        
    return np.array(feature_list), feature_names