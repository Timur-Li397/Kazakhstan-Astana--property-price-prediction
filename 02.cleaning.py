import pandas as pd


df = pd.read_csv("listings.csv")

print("Original shape:", df.shape)


columns_to_drop = [
    "url",
    "price_usd",
    "price_per_m2_usd",
    "parsed_at"
]

df = df.drop(columns=columns_to_drop)


print("\nTarget variable:")
print(df["price_kzt"].describe())


print("\nInvalid values:")

print("Price <= 0:", (df["price_kzt"] <= 0).sum())
print("Rooms <= 0:", (df["rooms"] <= 0).sum())
print("Area <= 0:", (df["total_area_m2"] <= 0).sum())
print("Year built <= 0:", (df["year_built"] <= 0).sum())


df = df[
    (df["price_kzt"] > 0)
    & (df["rooms"] > 0)
    & (df["total_area_m2"] > 0)
    & (df["year_built"] > 0)
].copy()


print("\nMissing values after basic cleaning:")
print(df.isna().sum())


print("\nCleaned shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


df.to_csv("listings_cleaned.csv", index=False)

print("\nSaved: listings_cleaned.csv")