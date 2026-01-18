
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np


def get_lag_values(df, col_name, lag_hours):
    lookup_table = df[['datetime','lon','lat',col_name]].sort_values('datetime')

    query_df = df[['datetime','lon','lat']].copy()
    query_df['target_time'] = query_df['datetime'] - pd.Timedelta(hours=lag_hours)
    query_df = query_df.sort_values('target_time')
    
    # Nearest Match
    merged = pd.merge_asof(
        query_df, lookup_table,
        left_on='target_time',
        right_on='datetime',
        by=['lon', 'lat'],
        direction='backward',
        suffixes=('', '_lag')
    )
    
    merged.index = query_df.index
    merged = merged.sort_index()
    return merged[col_name]


def GetStatistic(y_true, y_pred):
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    
    R2 = r2_score(y_true, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    MAE = mean_absolute_error(y_true, y_pred)
    MAPE = np.median(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    statistic = [R2, RMSE, MAE, MAPE]
    return statistic

def filter_data_columns(data):
    new_order = ['pm2.5', 'pm10', 'o3', 'no2', 'co', 'so2', 'time_sequence', 'lon', 'lat', 'year', 'month', 'day', 'hour' , 'week',
                 'prev_Radiance01', 'prev_Radiance02', 'prev_Radiance03', 'prev_Radiance04', 'prev_Radiance05', 'prev_Radiance06', 'prev_Raa', 'prev_SZA', 
                 'prev_VZA', 'prev2_VZA', 'prev_v10', 'prev2_v10', 'prev_u10', 'prev2_u10', 'altitude', 'prev_uvi', 'prev2_PS01', 'prev_PS02', 'prev2_PS02', 
                 'prev_RH01', 'prev2_RH01', 'prev_VP01', 'prev2_VP01', 'prev_GR01', 'prev2_GR01', 'prev_O3', 'prev2_O3', 'prev_NO2', 'prev2_NO2', 'prev_SO2',
                 'prev_AOD', 'prev2_AOD', 'prev_ALH', 'prev_CF', 'prev_R440', 'prev_SSA', 'prev24_v10', 'prev24_u10', 'prev24_PS01', 'prev24_PS02', 'prev24_RH01',
                 'prev24_VP01', 'prev24_O3', 'prev24_NO2', 'prev24_AOD', 'prev24_Radiance01', 'prev24_SZA', 'prev24_VZA']
    return data[new_order]

# ================================
# import the dataset
# ================================

origin_data = pd.read_csv('./data/example_data.csv')


# ================================
# feature engineering
# ================================
origin_data['datetime'] = (
    pd.to_datetime(origin_data['year'], format='%Y') + 
    pd.to_timedelta(origin_data['day'] - 1, unit='D') + 
    pd.to_timedelta(origin_data['hour'], unit='h'))

origin_data['week'] = (origin_data['datetime'].dt.dayofweek + 1).astype(str)
origin_data['time_sequence'] = origin_data['datetime'].rank(method='dense').astype(int)

prev_var = ['Radiance01', 'Radiance02', 'Radiance03', 'Radiance04', 'Radiance05', 'Radiance06',
    'Raa', 'SZA', 'VZA', 'v10', 'u10', 'uvi', 'PS01', 'PS02', 'RH01', 'VP01', 'GR01',
    'O3', 'NO2', 'SO2', 'AOD', 'ALH', 'CF', 'R440', 'SSA']

prev_var2 = ['VZA', 'v10', 'u10', 'PS01', 'PS02', 'RH01', 'VP01', 'GR01', 'O3', 'NO2', 'AOD']

prev_var24 = ['v10', 'u10', 'PS01', 'PS02', 'RH01', 'VP01', 'O3', 'NO2', 'AOD', 'Radiance01', 'SZA', 'VZA']

for var in prev_var:
    origin_data[f'prev_{var}'] = get_lag_values(origin_data, var, 1)

for var in prev_var2:
    origin_data[f'prev2_{var}'] = get_lag_values(origin_data, var, 2)

for var in prev_var24:
    origin_data[f'prev24_{var}'] = get_lag_values(origin_data, var, 24)  

# remove missing value
origin_data.dropna(inplace=True)

target_list = ['pm2.5', 'pm10', 'o3', 'no2', 'co', 'so2']

cat_feature = ['week']

features_list = ['lon', 'lat', 'year', 'month', 'day', 'hour' , 'week',
                 'prev_Radiance01', 'prev_Radiance02', 'prev_Radiance03', 'prev_Radiance04', 'prev_Radiance05', 'prev_Radiance06', 'prev_Raa', 'prev_SZA', 
                 'prev_VZA', 'prev2_VZA', 'prev_v10', 'prev2_v10', 'prev_u10', 'prev2_u10', 'altitude', 'prev_uvi', 'prev2_PS01', 'prev_PS02', 'prev2_PS02', 
                 'prev_RH01', 'prev2_RH01', 'prev_VP01', 'prev2_VP01', 'prev_GR01', 'prev2_GR01', 'prev_O3', 'prev2_O3', 'prev_NO2', 'prev2_NO2', 'prev_SO2',
                 'prev_AOD', 'prev2_AOD', 'prev_ALH', 'prev_CF', 'prev_R440', 'prev_SSA', 'prev24_v10', 'prev24_u10', 'prev24_PS01', 'prev24_PS02', 'prev24_RH01',
                 'prev24_VP01', 'prev24_O3', 'prev24_NO2', 'prev24_AOD', 'prev24_Radiance01', 'prev24_SZA', 'prev24_VZA']

metrics = {t: [] for t in target_list}

# output file
metric_names = ['R2', 'RMSE', 'MAE', 'MAPE']
for m_name in metric_names:
    filename = f'./metric/metric_{m_name}.csv'
    with open(filename, 'w') as f:
        header = ['date', 'day'] + target_list
        f.write(','.join(header) + '\n')

# 0701 -> 12/31
for start_index in range(182, 365 + 1, 1):
    print(f"Processing: {start_index}")

    # Train set
    train_data = origin_data[
        (origin_data['year'] == 2022) | 
        ((origin_data['year'] == 2023) & (origin_data['day'] < start_index))
    ]
    train_data = filter_data_columns(train_data)

    train_data = train_data.sort_values(by=['year', 'month', 'day', 'hour'])
    unique_time_points = train_data[['year', 'month', 'day', 'hour']].drop_duplicates()
    time_steps = len(unique_time_points)

    # Test set
    test_data = origin_data[(origin_data['year'] == 2023) & (origin_data['day'] == start_index)]
    if len(test_data) == 0: 
        continue
    test_data = filter_data_columns(test_data)
    test_data = test_data.sort_values(by=['year', 'month', 'day', 'hour'])
    unique_test_time_points = test_data[['year', 'month', 'day', 'hour']].drop_duplicates()
    test_time_steps = len(unique_test_time_points)

    X_train = train_data[features_list]
    y_train = train_data[target_list]

    X_test = test_data[features_list]
    y_test_true = test_data[target_list]

    # Build the multi-output Catboost model
    model = CatBoostRegressor(
        iterations = 150,            # 1400
        learning_rate = 0.4,
        loss_function = 'MultiRMSE', 
        boosting_type = 'Ordered',
        # task_type = 'GPU',
        random_seed = 42
    )
    model.fit(X_train, y_train, cat_features = [features_list.index('week')])

    # predict
    y_pred = model.predict(X_test)

    stats = {m: {} for m in metric_names}
    current_date = pd.to_datetime('2023-01-01') + pd.Timedelta(days=start_index - 1)
    date_str = current_date.strftime('%Y-%m-%d')

    # calculate the metrics
    for i, target_name in enumerate(target_list):
        fit_acc = np.zeros((1, 4))
        result_metric = GetStatistic(
            y_test_true[target_name].values.astype(float), 
            y_pred[:, i].astype(float))
        fit_acc[0, :] = result_metric
            
        for idx, m_name in enumerate(metric_names):
            stats[m_name][target_name] = result_metric[idx]
        metrics[target_name].append(fit_acc[0, :])

    # record
    for i in metric_names:
        row_data = [date_str, str(start_index)]
        for target in target_list:
            val = stats[i][target]
            row_data.append(f"{val:.6f}")

        with open(f'./metric/metric_{i}.csv', 'a') as f:
            f.write(','.join(row_data) + '\n')

