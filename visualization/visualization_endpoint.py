import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import math

# 데이터 최상위 경로
root_dir = './data'

# 마지막 몇 개의 점을 볼 것인지 설정 (예: 20개)
tail_length = 20 

def parse_trajectory_segments(filepath):
    segments = []
    current_seg = {'x': [], 'y': [], 'z': []}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                
                if parts[0] == 's':
                    if current_seg['x']:
                        segments.append(current_seg)
                        current_seg = {'x': [], 'y': [], 'z': []}
                    continue
                
                if parts[0] == 'r' and len(parts) > 6:
                    try:
                        coords = parts[6].split('/')
                        if len(coords) >= 3:
                            current_seg['x'].append(float(coords[0]))
                            current_seg['y'].append(float(coords[1]))
                            current_seg['z'].append(float(coords[2]))
                    except ValueError:
                        continue
                        
        if current_seg['x']:
            segments.append(current_seg)
            
    except Exception:
        pass
        
    return segments

subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
num_folders = len(subfolders)

if num_folders == 0:
    print("폴더를 찾을 수 없습니다.")
    exit()

cols = 3
rows = math.ceil(num_folders / cols)
fig = plt.figure(figsize=(cols * 5, rows * 5))

print(f"총 {num_folders}개 폴더의 '마지막 {tail_length}개' 구간을 시각화합니다.")

for i, folder_path in enumerate(subfolders):
    folder_name = os.path.basename(folder_path)
    file_list = glob.glob(os.path.join(folder_path, '*.txt'))
    
    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
    
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        segments = parse_trajectory_segments(file_path)
        
        p = None 
        
        for seg in segments:
            x_tail = seg['x'][-tail_length:]
            y_tail = seg['y'][-tail_length:]
            z_tail = seg['z'][-tail_length:]
            
            if not x_tail: continue

            if p is None:
                p = ax.plot(x_tail, y_tail, z_tail, label=file_name, linewidth=1.5, alpha=0.9)
                color = p[0].get_color()
                
                ax.scatter(x_tail[-1], y_tail[-1], z_tail[-1], color=color, s=20, marker='o')
            else:
                ax.plot(x_tail, y_tail, z_tail, color=color, linewidth=1.5, alpha=0.9)
                ax.scatter(x_tail[-1], y_tail[-1], z_tail[-1], color=color, s=20, marker='o')

    ax.set_title(f'Last {tail_length} pts: {folder_name}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if len(file_list) <= 10:
        ax.legend(fontsize='x-small')

plt.tight_layout()
plt.show()