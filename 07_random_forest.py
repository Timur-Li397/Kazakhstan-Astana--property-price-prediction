import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
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
# 3. Detect feature types
# ==========================================

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("Numerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


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
# 5. Random Forest model
# ==========================================

regressor = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ]
)


# ==========================================
# 6. Train model
# ==========================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# ==========================================
# 7. Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. Evaluation
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


# ==========================================
# 9. Results
# ==========================================

print("\nRandom Forest performance:")

print(f"MAE:  {mae:,.2f} KZT")
print(f"RMSE: {rmse:,.2f} KZT")
print(f"R²:   {r2:.4f}")


# ==========================================
# 10. Save predictions
# ==========================================

results = pd.DataFrame({
    "actual_price": y_test.values,
    "predicted_price": y_pred
})

results["absolute_error"] = (
    results["actual_price"]
    - results["predicted_price"]
).abs()

results.to_csv(
    "random_forest_predictions.csv",
    index=False
)

print("\nSaved: random_forest_predictions.csv")