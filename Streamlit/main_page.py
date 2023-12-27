
import streamlit as st
import clientserverUtils as csu
import userUtils

st.session_state.user : userUtils.SecurityOfficer = userUtils.SecurityOfficer()

#https://www.youtube.com/watch?v=eCbH2nPL9sU&ab_channel=CodingIsFun

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")


def logout(button_name: str, location: str = 'main', key: str = None):
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
            st.session_state['logout'] = True
    elif location == 'sidebar':
        if st.sidebar.button(button_name, key):
            st.session_state['logout'] = True

if csu.logged_in():
    logout('Logout', 'sidebar', key='unique_key')
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Some content')

