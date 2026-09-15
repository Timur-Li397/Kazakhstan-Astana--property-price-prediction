import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
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
# 5. Build model
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ==========================================
# 6. Train
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed.")


# ==========================================
# 7. Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. Metrics
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\nModel performance:")
print(f"MAE:  {mae:,.2f} KZT")
print(f"RMSE: {rmse:,.2f} KZT")
print(f"R²:   {r2:.4f}")


# ==========================================
# 9. Prediction statistics
# ==========================================

results = pd.DataFrame({
    "actual_price": y_test.values,
    "predicted_price": y_pred
})

results["error"] = (
    results["actual_price"]
    - results["predicted_price"]
)

results["absolute_error"] = (
    results["error"].abs()
)


print("\nPrediction statistics:")

print(
    results[
        [
            "actual_price",
            "predicted_price",
            "absolute_error"
        ]
    ].describe()
)


# ==========================================
# 10. Worst predictions
# ==========================================

print("\n10 largest prediction errors:")

worst_predictions = results.sort_values(
    "absolute_error",
    ascending=False
).head(10)

print(worst_predictions.to_string(index=False))


# ==========================================
# 11. Best predictions
# ==========================================

print("\n10 smallest prediction errors:")
best_predictions = results.sort_values(
    "absolute_error"
).head(10)

print(best_predictions.to_string(index=False))


# ==========================================
# 12. Actual vs predicted plot
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    results["actual_price"],
    results["predicted_price"],
    alpha=0.4
)

plt.xlabel("Actual price (KZT)")
plt.ylabel("Predicted price (KZT)")
plt.title("Actual vs Predicted Prices")

plt.tight_layout()
plt.show()


# ==========================================
# 13. Residual distribution
# ==========================================

plt.figure(figsize=(8, 6))

plt.hist(
    results["error"],
    bins=50
)

plt.xlabel("Prediction error (KZT)")
plt.ylabel("Number of listings")
plt.title("Distribution of Prediction Errors")

plt.tight_layout()
plt.show()