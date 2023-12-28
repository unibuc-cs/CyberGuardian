import streamlit as st
from userUtils import SecurityOfficer
from userUtils import SecurityOfficer
import pandas as pd


def logged_in() -> bool:
    return st.session_state.user.isLoggedIn()

def getName() -> str:
    return st.session_state.user.getName()

def login() -> bool:
    pass

def logout()-> bool:
    st.session_state.user = SecurityOfficer()

# Returns true if valid credetianls
def check_credentials() -> bool:
    # Check if user and passwords are ok. If they do mark as logged in.
    username = st.session_state['username']
    password = st.session_state['password']

    # TODO:
    # Retrieve from server the user details
    details = SecurityOfficer()
    details.name = "AlinRaresCiprian"
    st.session_state.user.login(details.__dict__)
    return True

# Clear and save
def clear_credentials_dataset(df):
    df = df[0:0]
    return df

def save_credentials_dataset(df, path):
    df.to_csv(path, index=False)

def RegisterError(msg : str):
    st.write(msg)