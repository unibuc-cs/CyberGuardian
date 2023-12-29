# Create
import os.path

import pandas
import pandas as pd
from userUtils import SecurityOfficer
from typing import Union

DEFAULT_CREDENTIALS_LOCATION = "data/cached_credentials.csv"

class CredentialsDB:
    def __init__(self):
        self.credentialsDF: pandas.DataFrame = None
        self.currentUser: SecurityOfficer = None
        self.reset()

    def reset(self):
        self.credentialsDF = None
        self.currentUser = None

    # Either load it from default path if exists, or creates and save for the first time
    def initialize(self):
        if os.path.exists(DEFAULT_CREDENTIALS_LOCATION):
            self.load_credentials_dataset()
        else:
            self.create_credentials_dataset()
            self.save_credentials_dataset()

    # Creates an empty dataframe
    def create_credentials_dataset(self) -> bool:
        assert self.credentialsDF is None, f"Dataset is NOT empty! {self.credentialsDF}"
        self.credentialsDF = pd.DataFrame({c: pd.Series(dtype=t) for c, t in
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
        return True

    def load_credentials_dataset(self, path: str = None) -> bool:
        assert self.credentialsDF is None or len(self.credentialsDF) == 0 , f"Dataset is NOT empty!. Clear it first if this is the intended workflow  {self.credentialsDF}"
        self.credentialsDF = pd.read_csv(path if path is not None else DEFAULT_CREDENTIALS_LOCATION)

    # Tests if the dataset is loaded / created
    def is_valid_credentials_dataset(self) -> bool:
        return self.credentialsDF is not None

    def getUserById(self, userName: str) -> Union[SecurityOfficer, None]:
        row = self.credentialsDF.loc[self.credentialsDF['username'] == userName].set_index('username').T.to_dict()
        #print(row)
        if len(row) == 0:
            return None
        else:
            res = SecurityOfficer()
            rowData = row[userName]
            for key, value in rowData.items():
                res.__setattr__(key, value)
            res.__setattr__('username', userName)
            #print(res)
            return res

    # Inserts a new user
    def insertNewUser(self, user: SecurityOfficer):
        assert self.credentialsDF is not None, "Dataframe is empty!"
        data = {name: value for name, value in user.__dict__.items()}
        self.credentialsDF.loc[len(self.credentialsDF.index)] = data

    # Clear and save
    def clear_credentials_dataset(self):
        self.credentialsDF = self.credentialsDF[0:0]

    def save_credentials_dataset(self, path: str = None):
        assert self.credentialsDF is not None, "Dataframe is empty!"
        self.credentialsDF.to_csv(path if path is not None else DEFAULT_CREDENTIALS_LOCATION, index=False)

    # Test if a user exists
    def userExists(self, userName: str) -> bool:
        row = self.credentialsDF.loc[self.credentialsDF['username'] == userName].set_index('username').T.to_dict()
        return len(row) > 0

    def getCurrentUser(self) -> SecurityOfficer:
        return self.currentUser

    def isLoggedIn(self) -> bool:
        return self.currentUser is not None

    def __repr__(self):
        return str(self.credentialsDF)

    # Checks the given credentials given and login if correct
    def tryLogin(self, userName: str, password: str) -> bool:
        row = self.credentialsDF.loc[self.credentialsDF['username'] == userName].set_index('username').T.to_dict()
        if row is not None and len(row) > 0 and row[userName]['password'] == password:
            self.currentUser = self.getUserById(userName)
            return True

        self.currentUser = None
        return False


