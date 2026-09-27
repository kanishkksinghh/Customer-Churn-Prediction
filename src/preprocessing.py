import pandas as pd
import numpy as np

def prepare_data(df):
    df = df.copy()

    df["Onboard_date"] = pd.to_datetime(
        df["Onboard_date"]
    )

    df["Onboard_Year"] = (
        df["Onboard_date"].dt.year
    )

    return df
    