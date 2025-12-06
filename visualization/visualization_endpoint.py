import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import math

# 데이터 최상위 경로
root_dir = './data'

def parse_trajectory_segments(filepath):
    """
    파일 내 's' 태그를 기준으로 궤적을 분리하여 (x, y, z) 리스트의 리스트로 반환
    """
    segments = []
    current_seg = {'x': [], 'y': [], 'z': []}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                
                # 's' 태그: 새로운 세그먼트 시작 (기존 세그먼트 저장)
                if parts[0] == 's':
                    if current_seg['x']:
                        segments.append(current_seg)
                        current_seg = {'x': [], 'y': [], 'z': []}
                    continue
                
                # 'r' 태그: 데이터 포인트 파싱
                if parts[0] == 'r' and len(parts) > 6:
                    try:
                        coords = parts[6].split('/')
                        if len(coords) >= 3:
                            current_seg['x'].append(float(coords[0]))
                            current_seg['y'].append(float(coords[1]))
                            current_seg['z'].append(float(coords[2]))
                    except ValueError:
                        continue
                        
        # 마지막 세그먼트 저장
        if current_seg['x']:
            segments.append(current_seg)
            
    except Exception:
        pass
        
    return segments

# 폴더 탐색
subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
num_folders = len(subfolders)

if num_folders == 0:
    print("폴더를 찾을 수 없습니다.")
    exit()

# 서브플롯 그리드 설정
cols = 3
rows = math.ceil(num_folders / cols)
fig = plt.figure(figsize=(cols * 5, rows * 5))

print(f"총 {num_folders}개 폴더의 '전체 궤적'을 시각화합니다.")

for i, folder_path in enumerate(subfolders):
    folder_name = os.path.basename(folder_path)
    file_list = glob.glob(os.path.join(folder_path, '*.txt'))
    
    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
    
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        segments = parse_trajectory_segments(file_path)
        
        # 파일별로 색상 고정을 위해 첫 세그먼트에서 plot 객체 생성 후 색상 추출
        p = None 
        
        for seg in segments:
            # [수정됨] tail_length 제한 없이 전체 포인트 사용
            x_pts = seg['x']
            y_pts = seg['y']
            z_pts = seg['z']
            
            if not x_pts: continue

            if p is None:
                # 첫 번째 세그먼트: 레이블 포함 및 색상 결정
                p = ax.plot(x_pts, y_pts, z_pts, label=file_name, linewidth=1.0, alpha=0.7)
                color = p[0].get_color()
                
                # 시작점(Green)과 끝점(Red) 표시
                ax.scatter(x_pts[0], y_pts[0], z_pts[0], c='green', s=20, marker='o') # Start
                ax.scatter(x_pts[-1], y_pts[-1], z_pts[-1], c='red', s=20, marker='x') # End
            else:
                # 이후 세그먼트: 동일 색상, 레이블 없음
                ax.plot(x_pts, y_pts, z_pts, color=color, linewidth=1.0, alpha=0.7)
                ax.scatter(x_pts[0], y_pts[0], z_pts[0], c='green', s=20, marker='o')
                ax.scatter(x_pts[-1], y_pts[-1], z_pts[-1], c='red', s=20, marker='x')

    ax.set_title(f'Full Trajectory: {folder_name}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 파일이 너무 많으면 범례 생략 (가독성 확보)
    if len(file_list) <= 10:
        ax.legend(fontsize='x-small')

plt.tight_layout()
plt.show()