import datetime

import pandas as pd
import yfinance as yf  # type: ignore
from nixtla import NixtlaClient

nixtla_client = NixtlaClient()
nixtla_client.validate_api_key()

TICKER = "BTC-USD"


def get_data(
    ticker: str = TICKER,
    end: datetime.date = datetime.date.today(),
    period: datetime.timedelta = datetime.timedelta(days=180),
) -> pd.DataFrame:
    start = end - period
    df = yf.download(ticker, start, end)
    return pd.DataFrame(
        {
            "unique_id": ticker,
            "ds": pd.to_datetime(df.index),
            "y": df["Close"][ticker],
        }
    )


def predict(data: pd.DataFrame, h: int = 30, freq: str = "D") -> pd.DataFrame:
    return nixtla_client.forecast(
        df=data, h=h, freq=freq, model="timegpt-1-long-horizon"
    )


def save_predictions(
    historic: pd.DataFrame, predictions: pd.DataFrame, output: str = "plot.png"
) -> None:
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    nixtla_client.plot(historic, predictions, ax=ax)
    ax.set_title(f"Predictions of {TICKER} as of {current_date}")
    ax.set_xlabel("")  # Remove x-axis title
    ax.set_ylabel("")  # Remove y-axis title
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )  # Format y-axis with $ and pretty print
    fig.savefig(output)


if __name__ == "__main__":
    data = get_data()
    predictions = predict(data)
    save_predictions(data, predictions)
