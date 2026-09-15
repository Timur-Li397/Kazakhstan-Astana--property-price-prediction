import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("listings.csv")



print("Shape:", df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nNumerical statistics:")
print(df.describe())


print("\nPrice quantiles:")
print(
    df["price_kzt"].quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
)


plt.figure(figsize=(10, 6))

plt.scatter(
    df["total_area_m2"],
    df["price_kzt"],
    alpha=0.3
)

plt.xlabel("Total area (m²)")
plt.ylabel("Price (KZT)")
plt.title("Apartment Price vs Total Area")

plt.show()



room_prices = (
    df.groupby("rooms")["price_kzt"]
    .agg(["mean", "median"])
)

room_prices.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.xlabel("Number of rooms")
plt.ylabel("Price (KZT)")
plt.title("Average and Median Apartment Price by Number of Rooms")
plt.xticks(rotation=0)
plt.legend(["Mean", "Median"])

plt.show()



district_prices = (
    df.groupby("district")["price_kzt"]
    .median()
    .sort_values(ascending=False)
)

print("\nMedian price by district:")
print(district_prices)



print("\nPotentially suspicious listings:")

print("\nLarge apartments:")
print(
    df.loc[
        df["total_area_m2"] > 300,
        ["price_kzt", "rooms", "total_area_m2", "district"]
    ]
)

print("\nApartments with many rooms:")
print(
    df.loc[
        df["rooms"] >= 8,
        ["price_kzt", "rooms", "total_area_m2", "district"]
    ]
)

print("\nInvalid floor relationships:")
print(
    df.loc[
        df["floor"] > df["total_floors"],
        ["price_kzt", "floor", "total_floors", "district"]
    ]
)



print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nPotential duplicate listings:")
print(
    df.duplicated(
        subset=[
            "price_kzt",
            "rooms",
            "total_area_m2",
            "district"
        ],
        keep=False
    ).sum()
)

print("\nDuplicate listing IDs:")
print(df["listing_id"].duplicated().sum())



print("\nCoordinate ranges:")
print(
    "Latitude:",
    df["latitude"].min(),
    "to",
    df["latitude"].max()
)

print(
    "Longitude:",
    df["longitude"].min(),
    "to",
    df["longitude"].max()
)

print("\nPotentially suspicious coordinates:")

print(
    df.loc[
        (df["latitude"] < 50)
        | (df["latitude"] > 52)
        | (df["longitude"] < 69)
        | (df["longitude"] > 73),
        [
            "listing_id",
            "price_kzt",
            "latitude",
            "longitude",
            "district"
        ]
    ]
)



print("\nPrice per m² statistics:")
print(df["price_per_m2_usd"].describe())

print("\nExtreme price per m²:")

print(
    df.loc[
        df["price_per_m2_usd"]
        > df["price_per_m2_usd"].quantile(0.99),
        [
            "price_kzt",
            "price_per_m2_usd",
            "rooms","total_area_m2",
            "district"
        ]
    ].head(20)
)


print("\nCity distribution:")
print(df["city"].value_counts(dropna=False))