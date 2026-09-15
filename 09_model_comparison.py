import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==========================================
# 1. Load data
# ==========================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train.drop(columns=["price_kzt"])
y_train = train["price_kzt"]

X_test = test.drop(columns=["price_kzt"])
y_test = test["price_kzt"]


# ==========================================
# 2. Remove ID / noisy features
# ==========================================

columns_to_drop = [
    "listing_id",
    "city",
    "address",
    "latitude",
    "longitude"
]

X_train = X_train.drop(
    columns=columns_to_drop,
    errors="ignore"
)

X_test = X_test.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# ==========================================
# 3. Feature types
# ==========================================

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ==========================================
# 4. Preprocessing
# ==========================================

numerical_transformer = SimpleImputer(
    strategy="median"
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
# 5. Define models
# ==========================================

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ==========================================
# 6. Train Linear Regression
# ==========================================

print("Training Linear Regression...")

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)

print("Linear Regression completed.")


# ==========================================
# 7. Train Random Forest
# ==========================================

print("\nTraining Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_predictions = (
    random_forest_model.predict(X_test)
)

print("Random Forest completed.")


# ==========================================
# 8. Calculate metrics
# ==========================================

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = mean_squared_error(
    y_test,
    linear_predictions
) ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


rf_mae = mean_absolute_error(
    y_test,
    random_forest_predictions
)

rf_rmse = mean_squared_error(
    y_test,
    random_forest_predictions
) ** 0.5

rf_r2 = r2_score(
    y_test,
    random_forest_predictions
)


# ==========================================
# 9. Create comparison table
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"],
    "MAE_KZT": [
        linear_mae,
        rf_mae
    ],
    "RMSE_KZT": [
        linear_rmse,
        rf_rmse
    ],
    "R2": [
        linear_r2,
        rf_r2
    ]
})


# ==========================================
# 10. Print comparison
# ==========================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    comparison.to_string(
        index=False,
        formatters={
            "MAE_KZT": "{:,.2f}".format,
            "RMSE_KZT": "{:,.2f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# ==========================================
# 11. Save comparison
# ==========================================

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nSaved: model_comparison.csv")


# ==========================================
# 12. Plot R² comparison
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["Model"],
    comparison["R2"]
)

plt.ylabel("R²")
plt.title("Model Comparison: R²")

plt.ylim(0, 1)

plt.tight_layout()
plt.show()