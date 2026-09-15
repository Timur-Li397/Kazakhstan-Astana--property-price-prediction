import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==========================================
# 1. Load train and test data
# ==========================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train.drop(columns=["price_kzt"])
y_train = train["price_kzt"]

X_test = test.drop(columns=["price_kzt"])
y_test = test["price_kzt"]


# ==========================================
# 2. Define feature types
# ==========================================

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ==========================================
# 3. Preprocessing
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
# 4. Build ML pipeline
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ==========================================
# 5. Train model
# ==========================================

print("Training Linear Regression...")

model.fit(X_train, y_train)

print("Training completed.")


# ==========================================
# 6. Make predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. Evaluate model
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


# ==========================================
# 8. Print metrics
# ==========================================

print("\nModel performance:")
print(f"MAE:  {mae:,.0f} KZT")
print(f"RMSE: {rmse:,.0f} KZT")
print(f"R²:   {r2:.4f}")