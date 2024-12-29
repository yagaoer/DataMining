import pandas as pd
import geopandas as gpd
import shapely.wkt
from shapely.geometry import Point, LineString
from sklearn.neighbors import NearestNeighbors
import numpy as np

# 1. 加载 GPS轨迹数据
traj_data = pd.read_csv('test_ori_traj.csv')

locations = traj_data['coordinates']
lon = []
lat = []
for every in locations:
    location = every.strip("[]").split(",")
    lon.append(location[0])
    lat.append(location[1])

# 处理 GPS 噪声：使用移动平均进行平滑
traj_data['latitude'] = pd.Series(lat).rolling(window=3, min_periods=1).mean()
traj_data['longitude'] = pd.Series(lon).rolling(window=3, min_periods=1).mean()

# 创建 GeoDataFrame
traj_data['geometry'] = traj_data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
gdf_traj = gpd.GeoDataFrame(traj_data, geometry='geometry')

# 3. 准备路网数据
road_network = gpd.read_file('road.shp')


# 4. 路网匹配函数
def match_to_road(traj, road):
    # 创建 NearestNeighbors 模型
    road_coords = np.array(list(road.geometry.apply(lambda geom: geom.coords[0])))
    nbrs = NearestNeighbors(n_neighbors=1).fit(road_coords)
    matched_lines = []
    for point in traj.geometry:
        dist, idx = nbrs.kneighbors(np.array([[point.x, point.y]]))
        matched_line = road.iloc[idx[0][0]].geometry
        matched_lines.append(matched_line)
    return matched_lines


# 进行路网匹配
gdf_traj['matched_line'] = match_to_road(gdf_traj, road_network)

# 5. 输出结果到 CSV
output_data = gdf_traj[['latitude', 'longitude']]
output_data['matched_line'] = gdf_traj['matched_line'].apply(lambda line: line.wkt)
output_data.to_csv('matched_traj.csv', index=False)

print("路网匹配完成，结果已保存为 matched_traj.csv")
