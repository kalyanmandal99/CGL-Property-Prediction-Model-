import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
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
N_SIM_POINTS = 50                    # points to sweep across the speed range

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
df_CGL_raw = pd.read_excel(r"Dataset\CGL Model Data_v2.xlsx", sheet_name="Main")
df_CGL_val_raw = pd.read_excel(r"Dataset\CGL Model Data_v2.xlsx", sheet_name="Validation")

df_CGL_raw.rename(columns={"TS": "UTS", "ELONGATION": "EL"}, inplace=True)
df_CGL_val_raw.rename(columns={"TS": "UTS", "ELONGATION": "EL"}, inplace=True)

print(df_CGL_raw.shape, df_CGL_val_raw.shape)

# Store results + fitted pipelines for later use (e.g. the speed simulation)
results = []
fitted_models = {}   # property -> {"model_name": pipeline}
train_features = {}  # property -> x_train dataframe (unencoded), for building simulation baselines

for property_ in PROPERTIES_TO_PREDICT:
    print("\n================ Building models for:", property_, "================")

    other_props = [p for p in PROPERTIES_TO_PREDICT if p != property_]

    # IMPORTANT: use .copy() so we never mutate the original loaded dataframes
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

    # ------------------------------------------------------------
    # Preprocessing: scale ALL numeric columns, one-hot encode the
    # categorical column. This replaces the manual encode/scale
    # steps that were previously dropping every numeric feature.
    # ------------------------------------------------------------
    numeric_cols = [c for c in x.columns if c != CATEGORICAL_COL]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [CATEGORICAL_COL]),
        ]
    )

    models = {
        "MLR": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=0),
        "XGBoost": XGBRegressor(random_state=0),
    }

    fitted_models[property_] = {}

    for model_name, estimator in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(x_train, y_train)

        y_pred = pipe.predict(x_test)
        y_train_pred = pipe.predict(x_train)
        y_val_pred = pipe.predict(x_val)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_pred)
        r2_val = r2_score(y_val, y_val_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        print(f"--- {model_name} ---")
        print(f"Train R2: {r2_train:.3f} | Test R2: {r2_test:.3f} | Val R2: {r2_val:.3f}")
        print(f"MAE: {mae:.2f} | MAPE: {mape:.2f}%")

        results.append({
            "Property": property_,
            "Model": model_name,
            "Train_R2": round(r2_train, 3),
            "Test_R2": round(r2_test, 3),
            "Val_R2": round(r2_val, 3),
            "MAE": round(mae, 2),
            "MAPE": round(mape, 2),
        })

        fitted_models[property_][model_name] = pipe

df_model_evaluation = pd.DataFrame(results)
print("\n\n===== Model evaluation summary =====")
print(df_model_evaluation.to_excel(r"Dataset\CGL Model Accuracy Matrics.xlsx"))


# ------------------------------------------------------------------
# SENSITIVITY SIMULATION
# For each ACTUAL TDC category: hold every other feature constant
# (at its median from the training data) and sweep Machine Speed
# across its observed range, to see how each property responds.
# ------------------------------------------------------------------
def simulate_speed_effect(pipe, x_train, categorical_col, speed_col,
                           property_name, model_name, n_points=N_SIM_POINTS):
    """Returns a long-format dataframe of simulated predictions and plots them."""

    categories = x_train[categorical_col].dropna().unique()

    # baseline = median for numeric cols, mode for any other categorical cols
    baseline = {}
    for col in x_train.columns:
        if col in (categorical_col, speed_col):
            continue
        if pd.api.types.is_numeric_dtype(x_train[col]):
            baseline[col] = x_train[col].median()
        else:
            baseline[col] = x_train[col].mode().iloc[0]

    speed_min, speed_max = x_train[speed_col].min(), x_train[speed_col].max()
    speed_range = np.linspace(speed_min, speed_max, n_points)

    all_sim_rows = []
    for cat in categories:
        sim_df = pd.DataFrame({speed_col: speed_range})
        for col, val in baseline.items():
            sim_df[col] = val
        sim_df[categorical_col] = cat

        # reorder columns to match training feature order
        sim_df = sim_df[x_train.columns]

        preds = pipe.predict(sim_df)
        sim_df["Predicted_" + property_name] = preds
        all_sim_rows.append(sim_df)

    sim_result = pd.concat(all_sim_rows, ignore_index=True)

    # plot
    plt.figure(figsize=(9, 6))
    for cat in categories:
        subset = sim_result[sim_result[categorical_col] == cat]
        plt.plot(subset[speed_col], subset["Predicted_" + property_name], label=str(cat))

    plt.xlabel(speed_col)
    plt.ylabel(f"Predicted {property_name}")
    plt.title(f"Effect of {speed_col} on {property_name} by {categorical_col} ({model_name})")
    plt.legend(title=categorical_col)
    plt.tight_layout()
    plt.savefig(f"speed_sensitivity_{property_name}_{model_name}.png", dpi=150)
    plt.show()

    return sim_result


# Run the simulation for each property using the RandomForest model
# (swap "RandomForest" for "MLR" or "XGBoost" if you prefer a different model)
SIM_MODEL = "RandomForest"

simulation_results = {}
for property_ in PROPERTIES_TO_PREDICT:
    pipe = fitted_models[property_][SIM_MODEL]
    x_train = train_features[property_]

    if SPEED_COL not in x_train.columns:
        print(f"WARNING: '{SPEED_COL}' not found in columns for {property_}. "
              f"Available columns: {list(x_train.columns)}")
        continue

    sim_result = simulate_speed_effect(
        pipe, x_train, CATEGORICAL_COL, SPEED_COL, property_, SIM_MODEL
    )
    simulation_results[property_] = sim_result