import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# --- Load the Saved Artifacts ---
try:
    model = joblib.load('xgboost_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    pca = joblib.load('pca.joblib')
    historical_df = pd.read_csv('engineered_features.csv', parse_dates=['Date'])
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Please run the `setup.py` script first to train the model and save the necessary files.")
    st.stop()

# --- App Layout ---
st.set_page_config(layout="wide")
st.title("🛒 Walmart Weekly Sales Forecaster")
st.write(
    "Enter the details for the week you want to predict. The app will use a trained XGBoost model to forecast the sales.")

# Store metadata mapping
store_metadata = {
    1: ("A", 151315),
    2: ("A", 202307),
    3: ("B", 37392),
    4: ("A", 205863),
    5: ("B", 34875),
    6: ("A", 202505),
    7: ("B", 70713),
    8: ("A", 155078),
    9: ("B", 125833),
    10: ("B", 126512),
    11: ("A", 207499),
    12: ("B", 112238),
    13: ("A", 219622),
    14: ("A", 200898),
    15: ("B", 123737),
    16: ("B", 57197),
    17: ("B", 93188),
    18: ("B", 120653),
    19: ("A", 203819),
    20: ("A", 203742),
    21: ("B", 140167),
    22: ("B", 119557),
    23: ("B", 114533),
    24: ("A", 203819),
    25: ("B", 128107),
    26: ("A", 152513),
    27: ("A", 204184),
    28: ("A", 206302),
    29: ("B", 93638),
    30: ("C", 42988),
    31: ("A", 203750),
    32: ("A", 203007),
    33: ("A", 39690),
    34: ("A", 158114),
    35: ("B", 103681),
    36: ("A", 39910),
    37: ("C", 39910),
    38: ("C", 39690),
    39: ("A", 184109),
    40: ("A", 155083),
    41: ("A", 196321),
    42: ("C", 39690),
    43: ("C", 41062),
    44: ("C", 39910),
    45: ("B", 118221),
}

# --- User Input in a Form ---
with st.form("prediction_form"):
    st.header("Input Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        store = st.selectbox("Select Store", options=sorted(historical_df['Store'].unique()))
        store_type, store_size = store_metadata[store]
        store_type_map = {'A': 1, 'B': 2, 'C': 3}

        dept = st.selectbox("Select Department", options=sorted(historical_df['Dept'].unique()))

    with col2:
        date = st.date_input("Select Date", datetime.date(2024, 12, 25))
        is_holiday = st.checkbox("Is it a Holiday week?")

    with col3:
        store_type_options = list(store_type_map.keys())
        store_type_selection = st.selectbox("Store Type", options=store_type_options)

    submitted = st.form_submit_button("Predict Sales")

# --- Prediction Logic ---
if submitted:
    # 1. Look up historical features
    # Find the closest historical data point to the selected date for that store/dept
    lookup_df = historical_df[
        (historical_df['Store'] == store) &
        (historical_df['Dept'] == dept)
        ].copy()

    if lookup_df.empty:
        st.warning("No historical data for this Store/Department. Prediction may be inaccurate.")
        # Use global averages as a fallback if needed
        historical_features = {
            'Weekly_Sales_lag_1': historical_df['Weekly_Sales_lag_1'].mean(),
            'Weekly_Sales_lag_52': historical_df['Weekly_Sales_lag_52'].mean(),
            'Weekly_Sales_ewma_4': historical_df['Weekly_Sales_ewma_4'].mean(),
            'Store_Size': historical_df['Store_Size'].mean()  # Add other numerical features
            ,
        }
    else:
        # Find the row with the date closest to the user's input
        lookup_df['date_diff'] = (lookup_df['Date'] - pd.to_datetime(date)).abs()
        closest_row = lookup_df.sort_values('date_diff').iloc[0]
        historical_features = closest_row.to_dict()
        st.info(f"Using historical features from the closest date: {closest_row['Date'].date()}")

    # 2. Create input DataFrame for the model
    input_data = {
        'Store': store,
        'Dept': dept,
        'IsHoliday': 1 if is_holiday else 0,
        'Store_Type': store_type_map[store_type],
        'Store_Size': store_size,
        'Year': date.year,
        'Month': date.month,
        'WeekOfYear': pd.to_datetime(date).isocalendar().week,
        'Weekly_Sales_lag_1': historical_features['Weekly_Sales_lag_1'],
        'Weekly_Sales_lag_52': historical_features['Weekly_Sales_lag_52'],
        'Weekly_Sales_ewma_4': historical_features['Weekly_Sales_ewma_4'],
        'Weekly_Sales_ewma_52': historical_features.get('Weekly_Sales_ewma_52', 0),  # add all features
        'Weekly_Sales_roll_mean_4': historical_features.get('Weekly_Sales_roll_mean_4', 0),
        'Weekly_Sales_roll_mean_52': historical_features.get('Weekly_Sales_roll_mean_52', 0),
        'Weeks_Since_Last_Holiday': historical_features.get('Weeks_Since_Last_Holiday', 0),
        'avg_sales_weekofyear_global_safe': historical_features.get('avg_sales_weekofyear_global_safe', 0),
        'avg_deptweek_past': historical_features.get('avg_deptweek_past', 0),
    }

    # Ensure all columns from training are present
    training_cols = preprocessor.get_feature_names_out()
    input_df = pd.DataFrame([input_data])

    # Ensure all columns expected by the preprocessor are present
    expected_cols = preprocessor.feature_names_in_

    # Add any missing columns with defaults
    for col in expected_cols:
        if col not in input_df:
            input_df[col] = 0  # or np.nan, depending on your pipeline strategy

    # Align order
    input_df_aligned = input_df[expected_cols]

    # 3. Preprocess the data
    processed_input = preprocessor.transform(input_df_aligned)
    pca_input = pca.transform(processed_input)

    # 4. Make prediction
    prediction = model.predict(pca_input)

    # 5. Display the result
    st.subheader("Forecast Result")
    st.metric(label="Predicted Weekly Sales", value=f"${prediction[0]:,.2f}")
