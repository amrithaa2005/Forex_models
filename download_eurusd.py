import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
from datetime import datetime, timedelta, timezone

# ==========================
# 🔐 ADD YOUR DETAILS HERE
# ==========================
ACCESS_TOKEN = "18b4a5ad070ec57ea803321291ef500f-79d1a1e88c0813e11c1e9d19ba00ee75"
ACCOUNT_ID = "101-001-38603273-001"

client = oandapyV20.API(access_token=ACCESS_TOKEN)

instrument = "EUR_USD"
granularity = "M15"

start_date = datetime(2022, 1, 1, tzinfo=timezone.utc)

# Use yesterday as safe upper boundary
end_date = datetime.now(timezone.utc) - timedelta(hours=1)

all_data = []

while start_date < end_date:
    next_date = start_date + timedelta(days=5)

    # Prevent requesting future time
    if next_date > end_date:
        next_date = end_date

    params = {
        "from": start_date.isoformat(),
        "to": next_date.isoformat(),
        "granularity": "M15",
        "price": "M"
    }

    r = instruments.InstrumentsCandles(
        instrument="EUR_USD",
        params=params
    )

    client.request(r)

    candles = r.response["candles"]

    for candle in candles:
        if candle["complete"]:
            all_data.append({
                "timestamp": candle["time"],
                "open": float(candle["mid"]["o"]),
                "high": float(candle["mid"]["h"]),
                "low": float(candle["mid"]["l"]),
                "close": float(candle["mid"]["c"]),
                "volume": candle["volume"]
            })

    start_date = next_date

df = pd.DataFrame(all_data)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

df.to_csv("EURUSD_M15.csv", index=False)

print("Download complete. Rows:", len(df))