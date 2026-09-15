import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("listings_cleaned.csv")

print("Dataset shape:", df.shape)


X = df.drop(columns=["price_kzt"])
y = df["price_kzt"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining set:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTest set:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


train = X_train.copy()
train["price_kzt"] = y_train

test = X_test.copy()
test["price_kzt"] = y_test

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("\nSaved:")
print("- train.csv")
print("- test.csv")