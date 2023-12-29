
import streamlit as st
import clientserverUtils as csu
import userUtils
from databaseUtils import CredentialsDB
from userUtils import SecurityOfficer

# Init the credentials database
if 'cdb' not in st.session_state:
    csu.checkCreateCredentialsDB()

#https://www.youtube.com/watch?v=eCbH2nPL9sU&ab_channel=CodingIsFun

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Dynabic demo from University of Bucharest!")
st.image("ubsigla.jpg", width=200)
st.sidebar.success("Select a functionality above.")


def logout(button_name: str, location: str = 'main', key: str = None)->True:
    """
    Creates a logout button.

    Parameters
    ----------
    button_name: str
        The rendered name of the logout button.
    location: str
        The location of the logout button i.e. main or sidebar.
    """
    if location not in ['main', 'sidebar']:
        raise ValueError("Location must be one of 'main' or 'sidebar'")
    if location == 'main':
        if st.button(button_name, key):
            csu.logout()
            return True
    elif location == 'sidebar':
        if st.sidebar.button(button_name, key):
            csu.logout()
            return True

    return False

placeholder = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()

if csu.logged_in():
    currentUser: SecurityOfficer = csu.getCurrentUser()
    st.sidebar.writer(f'### Hi, {currentUser.name}')
    st.sidebar.image(csu.getCachedImgPathForUsername(currentUser.username))

    placeholder3.empty()
    placeholder.write(f'### You are logged in, *{currentUser.name}*! \n\n')
    if placeholder2.button("Logout"):
        csu.logout()
        placeholder.empty()
        placeholder2.empty()
        placeholder3.write(f'### Please login !')
else:
    placeholder = st.empty()
    placeholder2 = st.empty()
    placeholder3.write(f'### Please login !')