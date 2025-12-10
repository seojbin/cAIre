import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import math


root_dir = './data'


def parse_trajectory_file(filepath):
    xs, ys, zs = [], [], []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if parts[0] == 'r' and len(parts) > 6:
                    try:
                        coords = parts[6].split('/')
                        if len(coords) >= 3:
                            xs.append(float(coords[0]))
                            ys.append(float(coords[1]))
                            zs.append(float(coords[2]))
                    except: continue
    except: pass
    return xs, ys, zs

subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
num_folders = len(subfolders)

if num_folders == 0:
    print("폴더를 찾을 수 없습니다.")
    exit()

cols = 3
rows = math.ceil(num_folders / cols)

fig = plt.figure(figsize=(cols * 5, rows * 5))

for i, folder_path in enumerate(subfolders):
    folder_name = os.path.basename(folder_path)
    file_list = glob.glob(os.path.join(folder_path, '*.txt'))
    
    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
    
    print(f"Plotting [{folder_name}]")
    
    for file_path in file_list:
        x, y, z = parse_trajectory_file(file_path)
        if x:
            ax.plot(x, y, z, label=os.path.basename(file_path), alpha=0.6, linewidth=0.8)

    ax.set_title(f'Folder: {folder_name}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if len(file_list) <= 5:
        ax.legend(fontsize='small')

plt.tight_layout()
plt.show()