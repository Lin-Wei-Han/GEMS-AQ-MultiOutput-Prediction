# Satellite-Grounded Fusion for Hourly Air Quality Prediction

The source code for the study "Satellite-Grounded Fusion for Hourly Air Quality Prediction: Insights from GEMS-Based Modeling in Taiwan." We use a multi-output CatBoost model in a rolling validation framework to predict hourly concentrations of six air pollutants: PM2.5, PM10, O3, NO2, CO, and SO2.

## Directory Structure

- `data/` : Directory for dataset. (Note: `example_data.csv` is the sample data for demo.)
- `metric/` : Directory for output evaluation metrics (R2, RMSE, MAE, MAPE).
- `train_rolling.py` : feature engineering, model training, and rolling validation.

## Usage


```Bash
pip install -r requirements.txt
```


```Bash
python train_rolling.py
```

## Output

The code generates CSV files in the `metric/` directory containing performance metrics for all six pollutants.

## Citation

If you use this code in your research, please cite the following paper:

Lin, W.-H., & Chan, T.-C. (2026). Satellite-Grounded Fusion for Hourly Air Quality Prediction: Insights from GEMS-Based Modeling in Taiwan. Submitted to [Journal Name] (Under Review).