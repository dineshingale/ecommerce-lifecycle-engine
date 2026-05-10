import pandas as pd

def engineer_features(df):
    # Synthetic Feature: Ratio of cart additions to sessions (no leakage — just arithmetic)
    df['add_to_cart_rate'] = df['total_adds'] / (df['total_sessions'] + 1)

    return df
