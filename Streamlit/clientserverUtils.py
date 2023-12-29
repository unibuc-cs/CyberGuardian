import streamlit as st

import databaseUtils
from userUtils import SecurityOfficer
from userUtils import SecurityOfficer
from databaseUtils import CredentialsDB
import pandas as pd

g_credentialsDB: CredentialsDB = None

def logged_in() -> bool:
    if g_credentialsDB is None:
        checkCreateCredentialsDB()
    return g_credentialsDB.isLoggedIn()

def getCurrentUser() -> SecurityOfficer:
    return g_credentialsDB.getCurrentUser()

def logout()-> bool:
    return st.session_state.cdb.logout()

# Returns true if valid credentials are valid and login
def tryLogin(userName: str, password: str) -> bool:
    # Check if user and passwords are ok. If they do mark as logged in.
    return g_credentialsDB.tryLogin(userName, password)

# Clear and save
def RegisterError(msg : str):
    st.markdown(f"## <span style='color:red'>{msg}</span>", unsafe_allow_html=True)
    #st.write("## " + msg)

def checkCreateCredentialsDB():
    global g_credentialsDB
    st.session_state.cdb = CredentialsDB()
    st.session_state.cdb.initialize()
    g_credentialsDB = st.session_state.cdb

def getCachedImgPathForUsername(username: str):
    res = f"data/{username}_cachedProfilePic.png"
    return res

def register_credentials(newUser: SecurityOfficer):

    bytes_data = newUser.picture.getvalue()
    # Check the type of bytes_data:
    # Should output: <class 'bytes'>
    #st.write(type(bytes_data))

    with open(getCachedImgPathForUsername(newUser.username), "wb") as f:
        f.write(bytes_data)

    g_credentialsDB.insertNewUser(newUser)
    g_credentialsDB.save_credentials_dataset()

def isValidNewUsername(new_username: str) -> bool:
    return not g_credentialsDB.userExists(new_username)

