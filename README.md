# 项目README

## 项目概述
本项目来自于BUAA数据挖掘大作业。
实现了四个主要任务，主要应用于GPS轨迹数据的处理与分析，涉及路网匹配、相似轨迹检索、车辆行驶时间估计以及轨迹终点位置预测。通过对GPS轨迹数据进行处理和建模，利用机器学习算法对轨迹数据进行分析与预测，帮助了解和分析交通轨迹的行为模式。

## 任务1：路网匹配
### 功能：
- 加载GPS轨迹数据，通过平滑算法（如移动平均）处理GPS噪声。
- 使用最邻近算法（Nearest Neighbors）将轨迹数据与路网数据匹配，将轨迹点与最近的道路连接线匹配。
- 输出路网匹配结果，生成CSV文件包含匹配后的轨迹经纬度及其对应的道路线几何信息。

### 输入：
- `ori_traj.csv`：原始轨迹数据文件，包含GPS轨迹的经纬度坐标。
- `road.shp`：道路网络数据文件（Shapefile格式），用于与轨迹数据进行匹配。

### 输出：
- `matched_traj.csv`：包含匹配后轨迹点与道路几何信息的CSV文件。

## 任务2：相似轨迹检索
### 功能：
- 通过动态时间规整（DTW）算法计算两条轨迹之间的相似度。
- 对任务轨迹数据与历史轨迹数据进行匹配，找到最相似的轨迹，并输出匹配结果。
- 可视化显示相似轨迹及其匹配的最优轨迹。

### 输入：
- `traj.csv`：历史轨迹数据文件，包含GPS轨迹及时间信息。
- `sim_task.csv`：待匹配的任务轨迹数据文件。

### 输出：
- `similarity_results.csv`：包含任务轨迹与匹配轨迹的相似度信息的CSV文件。
- 可视化图像：轨迹的可视化图像，显示任务轨迹与最佳匹配轨迹。

## 任务3：车辆行驶时间估计
### 功能：
- 基于轨迹的起点与终点的经纬度信息，以及轨迹点之间的时间间隔，使用机器学习模型（随机森林回归器）来估计车辆行驶时间（ETA）。
- 通过训练集和测试集进行模型训练与评估，输出预测的行驶时间，并计算模型的性能指标（MAE、MSE、R2分数）。
- 对预测结果进行可视化，分析预测的误差分布和特征重要性。

### 输入：
- `traj.csv`：历史轨迹数据，包含每个轨迹点的经纬度和时间信息。
- `eta_task.csv`：待预测的任务轨迹数据文件。

### 输出：
- `predicted_eta_results.csv`：包含预测结果的CSV文件，更新了任务轨迹的时间信息。
- 可视化图像：
  - 真实与预测行驶时间对比图。
  - 预测误差分布图。
  - 特征重要性图。

## 任务4：预测轨迹终点位置
### 功能：
- 使用历史轨迹的前几个点作为特征，通过随机森林回归模型预测轨迹的下一位置（终点坐标）。
- 训练和测试模型，输出轨迹预测的终点位置，并保存结果。
- 对模型的训练集与测试集进行评估，并计算预测误差。

### 输入：
- `traj.csv`：历史轨迹数据，包含每个轨迹点的经纬度和时间信息。
- `jump_task.csv`：待预测终点的轨迹数据。

### 输出：
- `predicted_jump.csv`：包含预测结果的CSV文件，更新了任务轨迹的终点位置。

## 依赖库
本项目使用以下Python库：
- `pandas`：用于数据加载、处理和分析。
- `numpy`：用于数值计算和数组操作。
- `geopandas`：用于处理地理数据，支持Shapefile格式。
- `shapely`：用于处理几何对象。
- `sklearn`：提供了机器学习模型和评估工具。
- `fastdtw`：实现了快速的动态时间规整算法。
- `matplotlib`：用于数据可视化。
- `datetime`：用于处理日期和时间。

### 安装依赖：
```bash
pip install pandas numpy geopandas shapely scikit-learn fastdtw matplotlib
```

## 文件结构
```
DM_2024_Dataset-master/
│          
├─task1
│  │  gcj_traj.csv
│  │  matched_traj.csv
│  │  ori_traj.csv
│  │  road.shp
│  │  task1.py
│  │  traj.csv
│  │  transfer.py
│  │  
│  └─task1_test
│          matched_traj_test.csv
│          road.cpg
│          road.csv
│          road.dbf
│          road.prj
│          road.shp
│          road.shx
│          task1_test.py
│          test_gcj_traj.csv
│          test_ori_traj.csv
│          test_traj.csv
│          
├─task2
│  │  similarity_results.csv
│  │  sim_task.csv
│  │  task2.py
│  │  traj.csv
│  │  trajectory_34234.0.png
│  │  trajectory_34235.0.png
│  │  trajectory_34236.0.png
│  │  
│  └─task2_test
│          similarity_results_test.csv
│          task2_test.py
│          test_sim.csv
│          test_traj.csv
│          trajectory_34234.0_test.png
│          trajectory_34235.0_test.png
│          trajectory_34236.0_test.png
│          
├─task3
│  │  eta_task.csv
│  │  features_importance_evaluate.png
│  │  predicted_error_distribution.png
│  │  predicted_eta_results.csv
│  │  predicted_travel_time.png
│  │  task3.py
│  │  traj.csv
│  │  
│  └─task3_test
│          features_importance_evaluate.png
│          predicted_error_distribution.png
│          predicted_eta_test.csv
│          predicted_travel_time.png
│          task3_test.py
│          test_eta.csv
│          test_traj.csv
│          
└─task4
    │  jump_task.csv
    │  predicted_jump.csv
    │  task4.py
    │  traj.csv
    │  
    └─task4_test
            predicted_jump_test.csv
            task4_test.py
            test_jump.csv
            test_traj.csv

每个任务下包含测试任务文件夹，方便运行和调试；部分任务做了可视化处理，见.png文件
部分数据说明：
├── ori_traj.csv               # 原始轨迹数据
├── road.shp                   # 道路网络数据（Shapefile格式）
├── traj.csv                   # 历史轨迹数据
├── sim_task.csv               # 待匹配轨迹数据
├── eta_task.csv               # 待预测车辆行驶时间的轨迹数据
├── jump_task.csv              # 待预测轨迹终点位置的任务数据
│
├── matched_traj.csv           # 路网匹配结果
├── similarity_results.csv     # 相似轨迹检索结果
├── predicted_eta_results.csv  # 预测的行驶时间结果
└── predicted_jump.csv         # 预测的轨迹终点位置结果
```
