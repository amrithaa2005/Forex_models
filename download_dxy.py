import yfinance as yf
import pandas as pd

dxy = yf.download(
    "DX-Y.NYB",
    start="2022-01-01",
    interval="1d",
    auto_adjust=True,
    progress=False
)

dxy = dxy.reset_index()
dxy = dxy.rename(columns={
    "Date": "date",
    "Open": "dxy_open",
    "High": "dxy_high",
    "Low": "dxy_low",
    "Close": "dxy_close",
    "Volume": "dxy_volume"
})

dxy.to_csv("DXY_D1.csv", index=False)

print("DXY daily downloaded:", len(dxy))