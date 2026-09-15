import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


train = pd.read_csv("train.csv")

X_train = train.drop(columns=["price_kzt"])
y_train = train["price_kzt"]


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


categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


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


print("Training Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


importances = (
    model
    .named_steps["regressor"]
    .feature_importances_
)


importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

 

importance_df["original_feature"] = (
    importance_df["feature"]
    .str.replace(r"^(num__|cat__)", "", regex=True)
    .str.split("_", n=1)
    .str[0]
)


grouped_importance = (
    importance_df
    .groupby("original_feature")["importance"]
    .sum()
    .sort_values(ascending=False)
)


print("\nFeature importance:")

print(
    grouped_importance
    .head(15)
    .to_string()
)


top_features = grouped_importance.head(15).sort_values()

plt.figure(figsize=(10, 7))

top_features.plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importances")

plt.tight_layout()
plt.show()



grouped_importance.to_csv(
    "feature_importance.csv",
    header=["importance"]
)

print("\nSaved: feature_importance.csv")