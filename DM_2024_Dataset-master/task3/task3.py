import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 数据加载
traj_data = pd.read_csv('traj.csv')
eta_task_data = pd.read_csv('eta_task.csv')

# 2. 格式化时间
traj_data['time'] = pd.to_datetime(traj_data['time'])
eta_task_data['time'] = pd.to_datetime(eta_task_data['time'], errors='coerce')


# 3. 提取特征
def prepare_test_features(test_data):
    test_features = []
    # 为每个测试轨迹构建特征
    grouped_test = test_data.groupby('trajectory_id')
    for trajectory_id, group in grouped_test:
        # group = group.sort_values(by='time')
        for i in range(len(group) - 1):
            # print(type(group.iloc[i].coordinates))
            start_location = group.iloc[i].coordinates.strip("[]").split(",")
            end_location = group.iloc[i + 1].coordinates.strip("[]").split(",")
            test_features.append([
                float(start_location[0]),  # 起点经度
                float(start_location[1]),  # 起点纬度
                float(end_location[0]),  # 终点经度
                float(end_location[1])  # 终点纬度
            ])
    return np.array(test_features)


def extract_eta_features(data):
    features = []
    labels = []
    grouped = data.groupby('trajectory_id')

    for trajectory_id, group in grouped:
        group = group.sort_values(by='time')
        for i in range(len(group) - 1):
            start_loc = list(map(float, group.iloc[i].coordinates.strip("[]").split(",")))
            end_loc = list(map(float, group.iloc[i + 1].coordinates.strip("[]").split(",")))
            start_time = group.iloc[i]['time']
            end_time = group.iloc[i + 1]['time']
            travel_time = (end_time - start_time).total_seconds()

            features.append(start_loc + end_loc)
            labels.append(travel_time)

    return np.array(features), np.array(labels)


X, y = extract_eta_features(traj_data)

# 4. 划分测试、运行集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_final = prepare_test_features(eta_task_data)

# 5. Train Model
eta_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
eta_model.fit(X_train, y_train)

# 6. 测试集预测 计算到达时间
eta_predictions = eta_model.predict(X_test)
eta_predictions_final = eta_model.predict(X_test_final)
print(eta_predictions)
# 先对eta_task_data按照'trajectory_id'进行分组，然后取每个组的第一条记录
grouped_records = eta_task_data.groupby('trajectory_id')
res = []

counter = 0
for trajectory_id, records in grouped_records:
    arr = eta_predictions_final[counter:counter + len(records) - 1]
    counter += len(records) - 1
    for i in range(1, len(records)):
        records.iloc[i, records.columns.get_loc('time')] = records.iloc[i - 1, records.columns.get_loc('time')] + \
                                                           pd.to_timedelta(int(arr[i - 1]), unit='seconds')
    res.append(records)
combined_df = pd.concat(res)
combined_df.to_csv('predicted_eta_results.csv', index=False)
print("ETA预测已完成，结果已保存到 'predicted_eta_results.csv'")

# 7. 评估模型
mae = mean_absolute_error(y_test, eta_predictions)
mse = mean_squared_error(y_test, eta_predictions)
r2 = r2_score(y_test, eta_predictions)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'R2 Score: {r2:.2f}')

# Cross-validation for robustness
cv_scores = cross_val_score(eta_model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f'Cross-validation MAE: {-np.mean(cv_scores):.2f}')

# 8. 可视化预测数据
plt.figure(figsize=(10, 6))
plt.scatter(y_test, eta_predictions, alpha=0.7)
plt.plot([0, max(y_test)], [0, max(y_test)], '--r')
plt.xlabel('True Travel Time (seconds)')
plt.ylabel('Predicted Travel Time (seconds)')
plt.title('True vs Predicted ETA')
plt.savefig(f'predicted_travel_time.png')

# 9. 分析预测偏差的分布，直观显示预测精度
residuals = y_test - eta_predictions
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k')
plt.xlabel('Prediction Error (seconds)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.savefig(f'predicted_error_distribution.png')

# 10. 评估特征的重要程度，起始经纬度和终点经纬度
feature_importance = eta_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(['start_lon', 'start_lat', 'end_lon', 'end_lat'], feature_importance)
plt.title('Feature Importance')
plt.savefig(f'features_importance_evaluate.png')
