import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

# 1. 加载数据
traj_data = pd.read_csv('traj.csv')
jump_task_data = pd.read_csv('jump_task.csv')

# 2. 格式化时间
traj_data['time'] = pd.to_datetime(traj_data['time'])
jump_task_data['time'] = pd.to_datetime(jump_task_data['time'], errors='coerce')

window_size = 5

# 3. 提取完整轨迹信息：计算每条轨迹的平均速度和时长
def extract_eta_features(data):
    features = []
    labels = []
    grouped = data.groupby('trajectory_id')
    for trajectory_id, group in grouped:
        # 排序数据点
        group = group.sort_values(by='time')

        if len(group) > window_size:
            for i in range(len(group) - window_size):
                feature = []
                for j in range(i, i + window_size):
                    location = group.iloc[j].coordinates.strip("[]").split(",")
                    feature.append(float(location[0]))
                    feature.append(float(location[1]))
                features.append(feature)
                next_point_location = group.iloc[i + window_size].coordinates.strip("[]").split(",")
                labels.append([float(next_point_location[0]), float(next_point_location[1])])
    return np.array(features), np.array(labels)

X, y = extract_eta_features(traj_data)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 进行预测并评估模型
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 评估训练集
mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)

# 评估测试集
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)

# 打印评估结果
print(f"Training MAE: {mae_train}")
print(f"Training MSE: {mse_train}")
print(f"Training RMSE: {rmse_train}")

print(f"Test MAE: {mae_test}")
print(f"Test MSE: {mse_test}")
print(f"Test RMSE: {rmse_test}")

# 7. 准备测试数据的特征
def prepare_test_features(test_data):
    test_features = []
    grouped_test = test_data.groupby('trajectory_id')

    for trajectory_id, group in grouped_test:
        if len(group) > window_size:
            feature = []
            for i in range(len(group) - window_size - 1, len(group) - 1):
                # Handle missing or invalid 'coordinates' values
                coordinates = group.iloc[i].coordinates
                if isinstance(coordinates, str):  # Check if coordinates is a string
                    location = coordinates.strip("[]").split(",")
                    feature.append(float(location[0]))
                    feature.append(float(location[1]))
                else:
                    # Handle invalid coordinates case, e.g., assign default values or skip
                    feature.append(0.0)  # Assign a default value, or handle appropriately
                    feature.append(0.0)
            test_features.append(feature)
        else:
            feature = []
            for i in range(window_size - len(group)):
                coordinates = group.iloc[0].coordinates
                if isinstance(coordinates, str):
                    location = coordinates.strip("[]").split(",")
                    feature.append(float(location[0]))
                    feature.append(float(location[1]))
                else:
                    feature.append(0.0)
                    feature.append(0.0)
            for i in range(len(group)):
                coordinates = group.iloc[i].coordinates
                if isinstance(coordinates, str):
                    location = coordinates.strip("[]").split(",")
                    feature.append(float(location[0]))
                    feature.append(float(location[1]))
                else:
                    feature.append(0.0)
                    feature.append(0.0)
            test_features.append(feature)

    return np.array(test_features)


X_test_final = prepare_test_features(jump_task_data)

# 8. 进行预测
jump_predictions = model.predict(X_test_final)

# 9. 将预测结果更新到jump_task_data
grouped_records = jump_task_data.groupby('trajectory_id')
res = []
counter = 0
for trajectory_id, records in grouped_records:
    jump_prediction = jump_predictions[counter]
    records.iloc[-1, records.columns.get_loc('coordinates')] = "[" + str(jump_prediction[0]) + ", " +  str(jump_prediction[1]) + "]"
    res.append(records)
    counter += 1

# 合并预测结果并保存
combined_df = pd.concat(res)
combined_df.to_csv('predicted_jump.csv', index=False)

print("Jump预测已完成，结果已保存到 'predicted_jump.csv'")
