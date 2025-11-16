#3D 궤적 데이터 리스트를 2D 배열로 변환하는 코드
import numpy as np
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
        'ratio'
    ]

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
        if len(traj) > 2:
            try:
                pca = PCA(n_components=1)
                pca.fit(traj)
                # 1.0에 가까우면 직선, 0.5에 가까우면 원
                ratio = pca.explained_variance_ratio_[0]
            except Exception:
                ratio = 0.0 

        # 모든 특성을 하나의 리스트로 결합
        features = np.concatenate([
            means, stds, mins, maxs,
            [length, disp, length_disp],
            diff,
            bounding_box,
            [ratio]
        ])
        feature_list.append(features)
        
    return np.array(feature_list), feature_names
