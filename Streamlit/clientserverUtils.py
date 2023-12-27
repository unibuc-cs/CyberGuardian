import streamlit as st
from userUtils import SecurityOfficer


def logged_in() -> bool:
    return st.session_state.user.isLoggedIn()

def getName() -> str:
    return st.session_state.user.getName()

def login() -> bool:
    pass

def logout()-> bool:
    st.session_state.user = SecurityOfficer()


def RegisterError(msg : str):
    st.write(msg)