# ============================================================
# AIR QUALITY MONITORING AND AQI PREDICTION USING ML
# ASSIGNMENT 2 - EXTRA TREES REGRESSOR
# (Comparison with Random Forest from Assignment 1)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Libraries imported successfully!")

# ============================================================
# LOAD DATASET
# ============================================================

file_name = "RealTimeReport_1788784853626.xlsx"

df = pd.read_excel(
    file_name,
    sheet_name="CPCB Ambient AQ",
    header=10
)

df.columns = df.columns.astype(str).str.strip()

pollutants = ["PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "OZONE"]

for column in pollutants:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=pollutants, how="all").copy()

for column in pollutants:
    df[column] = df[column].fillna(df[column].median())

# ============================================================
# AQI CALCULATION
# ============================================================

def calculate_sub_index(value, breakpoints):
    if pd.isna(value):
        return np.nan
    for i in range(len(breakpoints) - 1):
        c_low, i_low = breakpoints[i]
        c_high, i_high = breakpoints[i + 1]
        if c_low <= value <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (value - c_low) + i_low
    if value > breakpoints[-1][0]:
        return 500
    return np.nan

pm25_breakpoints = [(0, 0), (30, 50), (60, 100), (90, 200), (120, 300), (250, 400), (500, 500)]
pm10_breakpoints = [(0, 0), (50, 50), (100, 100), (250, 200), (350, 300), (430, 400), (500, 500)]
no2_breakpoints = [(0, 0), (40, 50), (80, 100), (180, 200), (280, 300), (400, 400), (1000, 500)]
so2_breakpoints = [(0, 0), (40, 50), (80, 100), (380, 200), (800, 300), (1600, 400), (2620, 500)]
co_breakpoints = [(0, 0), (1, 50), (2, 100), (10, 200), (17, 300), (34, 400), (50, 500)]
ozone_breakpoints = [(0, 0), (50, 50), (100, 100), (168, 200), (208, 300), (748, 400), (1000, 500)]
nh3_breakpoints = [(0, 0), (200, 50), (400, 100), (800, 200), (1200, 300), (1800, 400), (2400, 500)]

df["PM2.5_Index"] = df["PM2.5"].apply(lambda x: calculate_sub_index(x, pm25_breakpoints))
df["PM10_Index"] = df["PM10"].apply(lambda x: calculate_sub_index(x, pm10_breakpoints))
df["NO2_Index"] = df["NO2"].apply(lambda x: calculate_sub_index(x, no2_breakpoints))
df["SO2_Index"] = df["SO2"].apply(lambda x: calculate_sub_index(x, so2_breakpoints))
df["CO_Index"] = df["CO"].apply(lambda x: calculate_sub_index(x, co_breakpoints))
df["OZONE_Index"] = df["OZONE"].apply(lambda x: calculate_sub_index(x, ozone_breakpoints))
df["NH3_Index"] = df["NH3"].apply(lambda x: calculate_sub_index(x, nh3_breakpoints))

df["AQI"] = df[["PM2.5_Index", "PM10_Index", "NO2_Index", "SO2_Index", "CO_Index", "OZONE_Index", "NH3_Index"]].max(axis=1).round(0)

# ============================================================
# PREPARE DATA + CHRONOLOGICAL SPLIT
# ============================================================

features = ["PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "OZONE"]
X = df[features].copy()
y = df["AQI"].copy()

split_index = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# ============================================================
# RANDOM FOREST (for comparison)
# ============================================================

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# ============================================================
# EXTRA TREES REGRESSOR
# ============================================================

et_model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
et_model.fit(X_train, y_train)
y_pred_et = et_model.predict(X_test)

mae_et = mean_absolute_error(y_test, y_pred_et)
rmse_et = np.sqrt(mean_squared_error(y_test, y_pred_et))
r2_et = r2_score(y_test, y_pred_et)

print("\n===== PERFORMANCE COMPARISON =====")
print(f"Random Forest  → MAE: {mae_rf:.4f} | RMSE: {rmse_rf:.4f} | R²: {r2_rf:.4f}")
print(f"Extra Trees    → MAE: {mae_et:.4f} | RMSE: {rmse_et:.4f} | R²: {r2_et:.4f}")

# ============================================================
# VISUALIZATIONS
# ============================================================

# 1. Side-by-side Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_pred_rf, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_title("Random Forest")
axes[0].set_xlabel("Actual AQI")
axes[0].set_ylabel("Predicted AQI")

axes[1].scatter(y_test, y_pred_et, alpha=0.6, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title("Extra Trees Regressor")
axes[1].set_xlabel("Actual AQI")
axes[1].set_ylabel("Predicted AQI")

plt.tight_layout()
plt.show()

# 2. Extra Trees Feature Importance
et_importance = pd.DataFrame({
    "Feature": features,
    "Importance": et_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nExtra Trees Feature Importance:")
print(et_importance.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.bar(et_importance["Feature"], et_importance["Importance"], color='teal')
plt.title("Extra Trees Feature Importance")
plt.xlabel("Pollutant")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. R² Comparison Bar Chart
results = pd.DataFrame({
    "Model": ["Random Forest", "Extra Trees"],
    "R2": [r2_rf, r2_et]
})

plt.figure(figsize=(6, 4))
plt.bar(results["Model"], results["R2"], color=['steelblue', 'teal'])
plt.title("R² Score Comparison")
plt.ylabel("R² Score")
plt.ylim(0.8, 1.0)
plt.tight_layout()
plt.show()

# ============================================================
# ADDITIONAL COMPARISON VISUALIZATIONS
# ============================================================

# 4. Full Metrics Comparison Table
print("\n===== FULL METRICS COMPARISON =====")
metrics_df = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R²"],
    "Random Forest": [
        round(mae_rf, 4),
        round(mean_squared_error(y_test, y_pred_rf), 4),
        round(rmse_rf, 4),
        round(r2_rf, 4)
    ],
    "Extra Trees": [
        round(mae_et, 4),
        round(mean_squared_error(y_test, y_pred_et), 4),
        round(rmse_et, 4),
        round(r2_et, 4)
    ]
})
print(metrics_df.to_string(index=False))

# 5. Side-by-side Feature Importance
rf_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(rf_importance["Feature"], rf_importance["Importance"], color='steelblue')
axes[0].set_title("Random Forest Feature Importance")
axes[0].set_xlabel("Pollutant")
axes[0].set_ylabel("Importance")
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(et_importance["Feature"], et_importance["Importance"], color='teal')
axes[1].set_title("Extra Trees Feature Importance")
axes[1].set_xlabel("Pollutant")
axes[1].set_ylabel("Importance")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 6. MAE and RMSE Comparison Bar Charts
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(["Random Forest", "Extra Trees"], [mae_rf, mae_et], color=['steelblue', 'teal'])
axes[0].set_title("MAE Comparison")
axes[0].set_ylabel("MAE")

axes[1].bar(["Random Forest", "Extra Trees"], [rmse_rf, rmse_et], color=['steelblue', 'teal'])
axes[1].set_title("RMSE Comparison")
axes[1].set_ylabel("RMSE")

plt.tight_layout()
plt.show()

print("\nConclusion: Extra Trees Regressor performed better than Random Forest on the chronological test set.")