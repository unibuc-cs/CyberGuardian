# Create

import pandas as pd
def create_credentials_dataset():
    df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                       {'name': 'str',
                        'username': 'str',
                        'password': 'str',
                        'birthday' : 'datetime64[ns]',
                        'expertise': 'int',
                        'preference': 'int',
                        'politely': 'int',
                        'emojis': 'int',
                        'motivation_factor': 'float',
                        'can_be_tricked_out_factor': 'float',
                        'intentional_damage_factor': 'float',
                        'correct_teamwork': 'float'
                        }.items()})
    return df

# Clear and save
def clear_credentials_dataset(df):
    df = df[0:0]
    return df

def save_credentials_dataset(df, path):
    df.to_csv(path, index=False)

