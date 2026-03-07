import pandas as pd

df = pd.read_csv("EURUSD_M15.csv")

print("Total rows:", len(df))
print("\nFirst 3 rows:")
print(df.head(3))

print("\nLast 3 rows:")
print(df.tail(3))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate timestamps:")
print(df.duplicated(subset="timestamp").sum())

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

print("\nTime interval check:")
print(df["timestamp"].diff().value_counts().head())
exit()