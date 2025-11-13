#3D 궤적 데이터 리스트를 2D 배열로 변환하는 코드
import numpy as np
def extractfeatures(trajectories):

    
    feature_list = []
    feature_names = [
        'X_mean', 'Y_mean', 'Z_mean',
        'X_std', 'Y_std', 'Z_std',
        'X_min', 'Y_min', 'Z_min',
        'X_max', 'Y_max', 'Z_max',
        'total_length', 'total_displacement',
        'length_disp_ratio'
    ]

    for traj in trajectories:
        #각 축의 기본 통계
        means = np.mean(traj, axis=0)
        stds = np.std(traj, axis=0)
        mins = np.min(traj, axis=0)
        maxs = np.max(traj, axis=0)
        
        # 궤적의 전체 길이 (모든 이동 거리 합)
        step_distances = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        total_length = np.sum(step_distances)

        # 궤적의 총 변위 (시작점과 끝점 사이의 거리)
        start_point = traj[0]
        end_point = traj[-1]
        total_displacement = np.linalg.norm(end_point - start_point)
        
        # 변위 대비 길이 비율
        # (0으로 나누기 방지)
        if total_displacement < 1e-6:
            length_disp_ratio = total_length
        else:
            length_disp_ratio = total_length / total_displacement

        # 모든 특성 결합
        features = np.concatenate([
            means, stds, mins, maxs,
            [total_length, total_displacement, length_disp_ratio]
        ])
        feature_list.append(features)
        
    # (샘플 수, 15) 형태의 2D 배열 반환
    return np.array(feature_list), feature_names