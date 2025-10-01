import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import xgboost as xgb
import joblib

# --- 1. Load and Prepare Your Data ---
# Make sure your original data file is in the same folder
try:
    df = pd.read_csv('engineered_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please place your training data in the same directory.")
    exit()

print("Starting feature engineering...")
# This should be the full feature engineering from your notebook
df = df.sort_values(by=['Store', 'Dept', 'Date'])
grp = df.groupby(['Store', 'Dept'])

df['Weekly_Sales_lag_1'] = grp['Weekly_Sales'].transform(lambda s: s.shift(1))
df['Weekly_Sales_lag_52'] = grp['Weekly_Sales'].transform(lambda s: s.shift(52))
df['Weekly_Sales_ewma_4'] = grp['Weekly_Sales'].transform(lambda s: s.shift(1).ewm(span=4).mean())
# Add any other features your final model needs...

df['Year'] = df['Date'].dt.year
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Drop rows with NaNs created by lags/ewma
df.dropna(inplace=True)

print("Feature engineering complete.")

# --- 2. Define Features and Preprocessing Pipeline ---
# Use all features except the target and the original date
X = df.drop(columns=['Weekly_Sales', 'Date'])
y = df['Weekly_Sales']

numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = ['Store', 'Dept', 'Store_Type', 'IsHoliday']
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ])

# --- 3. Fit Preprocessor and PCA ---
X_processed = preprocessor.fit_transform(X)

N_COMPONENTS = 115
pca = PCA(n_components=N_COMPONENTS)
pca.fit(X_processed)

# --- 4. Train Final XGBoost Model ---
# We'll use the best parameters you found during tuning
best_params = {
    'subsample': 0.7,
    'n_estimators': 1000,
    'max_depth': 3,
    'learning_rate': 0.01,
    'colsample_bytree': 1.0,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'random_state': 42
}

final_model = xgb.XGBRegressor(**best_params)

# We need to transform the data before fitting
X_pca = pca.transform(X_processed)

print("Training the final XGBoost model on all data...")
final_model.fit(X_pca, y)

# --- 5. Save the Artifacts ---
joblib.dump(final_model, 'xgboost_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(pca, 'pca.joblib')
# Save the engineered dataframe so the app can look up historical features
df.to_csv('engineered_features.csv', index=False)

print("\n✅ Setup complete! Model, preprocessor, PCA, and data have been saved.")
