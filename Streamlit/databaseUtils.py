# Create

import pandas as pd
from userUtils import SecurityOfficer
from typing import Union

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

def getUserById(userName: str) -> Union[SecurityOfficer, None]:
    row = df.loc[df['username'] == userName].set_index('username').T.to_dict()
    if len(row) == 0:
        return None
    else:
        res = SecurityOfficer()
        rowData = row[userName]
        for key, value in rowData.items():
            res.__setattr__(key, value)
        res.__setattr__('username', userName)
        return res

# Clear and save
def clear_credentials_dataset(df):
    df = df[0:0]
    return df

def save_credentials_dataset(df, path):
    df.to_csv(path, index=False)

