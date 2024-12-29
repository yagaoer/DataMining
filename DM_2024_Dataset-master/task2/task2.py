from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载数据
traj_data = pd.read_csv('traj.csv')
sim_task_data = pd.read_csv('sim_task.csv')


# 提取轨迹的函数
def get_trajectory(data, trajectory_id):
    traj = data[data['trajectory_id'] == trajectory_id].sort_values(by='time')
    coordinates = []
    for _, row in traj.iterrows():
        location = row['coordinates'].strip("[]").split(",")
        coordinates.append([float(location[0]), float(location[1])])

    coordinates = np.array(coordinates)
    scaler = StandardScaler()
    coordinates = scaler.fit_transform(coordinates)

    if len(coordinates) > 3:
        cs = CubicSpline(np.arange(len(coordinates)), coordinates, axis=0)
        interpolated_points = cs(np.linspace(0, len(coordinates) - 1, 100))
        return interpolated_points
    return coordinates


# DTW距离计算

def dtw_distance(s1, s2):
    distance, _ = fastdtw(s1, s2, dist=lambda x, y: np.linalg.norm(x - y))
    return distance


# 计算相似性
similarities = []
sim_grouped = sim_task_data.groupby('trajectory_id')
traj_grouped = traj_data.groupby('trajectory_id')
trajs = []

for trajectory_id, group in traj_grouped:
    traj = get_trajectory(traj_data, trajectory_id)
    trajs.append([trajectory_id, traj])

for trajectory_id, group in sim_grouped:
    sim_trajectory = get_trajectory(sim_task_data, trajectory_id)
    best_match = None
    best_distance = float('inf')
    for every_traj in trajs:
        distance = dtw_distance(sim_trajectory, every_traj[1])
        if distance < best_distance:
            best_distance = distance
            best_match = every_traj[0]
    similarities.append([trajectory_id, best_match, best_distance])

similarity_results = pd.DataFrame(similarities,
                                  columns=['sim_task_trajectory_id', 'traj_best_match_trajectory_id', 'dtw_distance'])
similarity_results.to_csv('similarity_results.csv', index=False)

# 可视化轨迹并保存图片
for index, row in similarity_results.iterrows():
    sim_trajectory_id = row['sim_task_trajectory_id']
    best_match_id = row['traj_best_match_trajectory_id']

    sim_trajectory = get_trajectory(sim_task_data, sim_trajectory_id)
    best_trajectory = get_trajectory(traj_data, best_match_id)

    plt.figure(figsize=(10, 6))
    plt.plot(sim_trajectory[:, 1], sim_trajectory[:, 0], color='blue', label=f'Sim Task {sim_trajectory_id}')
    plt.plot(best_trajectory[:, 1], best_trajectory[:, 0], color='red', linestyle='--',
             label=f'Best Match {best_match_id}')
    plt.scatter(sim_trajectory[0, 1], sim_trajectory[0, 0], c='green', marker='o', label='Start Point')
    plt.scatter(sim_trajectory[-1, 1], sim_trajectory[-1, 0], c='black', marker='x', label='End Point')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Trajectory Visualization - Sim Task {sim_trajectory_id}')
    plt.legend()
    plt.savefig(f'trajectory_{sim_trajectory_id}.png')
    plt.close()


