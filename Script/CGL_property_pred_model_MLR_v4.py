import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIG - adjust these to match your actual column names
# ------------------------------------------------------------------
CATEGORICAL_COL = "ACTUAL TDC"
SPEED_COL = "FURNACE SPEED"          # <-- change if your speed column has a different name
PROPERTIES_TO_PREDICT = ["UTS", "YS", "EL"]

SELECTED_TDC = "GPK001"                   # <-- set this to the ONE Actual TDC value you want to test
N_SIM_POINTS = 50                    # points to sweep across the speed range

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
df_CGL_raw = pd.read_excel(r"Dataset\CGL Model Data_v2.xlsx", sheet_name="Main")
df_CGL_val_raw = pd.read_excel(r"Dataset\CGL Model Data_v2.xlsx", sheet_name="Validation")
df_CGL_test_raw = pd.read_excel(r"Dataset\CGL Model Data_v2.xlsx", sheet_name="Test v1")



df_CGL_raw.rename(columns={"TS": "UTS", "ELONGATION": "EL"}, inplace=True)
df_CGL_val_raw.rename(columns={"TS": "UTS", "ELONGATION": "EL"}, inplace=True)
df_CGL_test_raw.rename(columns={"TS": "UTS", "ELONGATION": "EL"}, inplace=True)


fitted_models = {}   # property -> fitted MLR pipeline
train_features = {}  # property -> x_train dataframe (unencoded)

for property_ in PROPERTIES_TO_PREDICT:
    print("\n=== Building MLR model for:", property_, "===")

    other_props = [p for p in PROPERTIES_TO_PREDICT if p != property_]

    df = df_CGL_raw.copy().drop(columns=other_props)
    df_val = df_CGL_val_raw.copy().drop(columns=other_props)

    x = df.drop(columns=[property_])
    y = df[property_]

    x_val = df_val.drop(columns=[property_])
    y_val = df_val[property_]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=0
    )
    train_features[property_] = x_train

    numeric_cols = [c for c in x.columns if c != CATEGORICAL_COL]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [CATEGORICAL_COL]),
    ])

    mlr_pipe = Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])
    mlr_pipe.fit(x_train, y_train)

    r2_test = r2_score(y_test, mlr_pipe.predict(x_test))
    r2_val = r2_score(y_val, mlr_pipe.predict(x_val))
    mae = mean_absolute_error(y_test, mlr_pipe.predict(x_test))
    print(f"Test R2: {r2_test:.3f} | Val R2: {r2_val:.3f} | MAE: {mae:.2f}")

    fitted_models[property_] = mlr_pipe

    
df_prediction = df_CGL_test_raw.copy()

for property_ in PROPERTIES_TO_PREDICT:

    model = fitted_models[property_]

    x_new = df_prediction.drop(columns=PROPERTIES_TO_PREDICT, errors="ignore")

    df_prediction[property_ + "_Predicted"] = model.predict(x_new)

print(df_prediction)