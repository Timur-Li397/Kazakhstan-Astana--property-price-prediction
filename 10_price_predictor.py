import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==========================================
# 1. Load training data
# ==========================================

train = pd.read_csv("train.csv")

X_train = train.drop(columns=["price_kzt"])
y_train = train["price_kzt"]


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
# 5. Random Forest
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
# 6. Train final model
# ==========================================

print("Training final Random Forest model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ==========================================
# 7. Save model
# ==========================================

joblib.dump(
    model,
    "random_forest_model.pkl"
)

print("\nSaved: random_forest_model.pkl")