import pandas as pd

def resample_ticks(df: pd.DataFrame, timeframe: str = "1S") -> pd.DataFrame:
    """
    Aggregate raw ticks into time bars by symbol.
    timeframe examples: '1S','5S','15S','30S','1T','5T','15T'
    price = last trade price in the bar
    volume = sum of sizes
    trades = number of ticks in the bar
    """
    df = df.copy()
    df = df[['ts', 'symbol', 'price', 'size']].dropna()
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values('ts')

    # set index for resample
    df = df.set_index('ts')

    bars = (df
            .groupby('symbol')
            .resample(timeframe)
            .agg(price=('price', 'last'),
                 volume=('size', 'sum'),
                 trades=('price', 'count'))
            .reset_index())

    # drop empty bars
    bars = bars.dropna(subset=['price'])
    return bars

