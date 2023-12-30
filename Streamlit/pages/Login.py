
import streamlit as st
from userUtils import SecurityOfficer
import clientserverUtils as csu
from databaseUtils import CredentialsDB

if 'cdb' not in st.session_state:
    csu.checkCreateCredentialsDB()


def check_cookie():
    pass
#
# def login(login_form, location: str = 'main') -> bool:
#     """
#     Creates a login widget.
#     """
#     if location not in ['main', 'sidebar']:
#         raise ValueError("Location must be one of 'main' or 'sidebar'")
#     if not csu.logged_in():
#         check_cookie()
#         if not csu.logged_in():
#             """
#             if location == 'main':
#                 login_form = st.form('Login')
#             elif location == 'sidebar':
#                 login_form = st.sidebar.form('Login')
#             """
#
#             login_form.subheader("Login")
#             username = st.text_input('Username').lower()
#             st.session_state['username'] = username
#             password = st.text_input('Password', type='password')
#             st.session_state['password'] = username
#
#             if st.form_submit_button('Login'):
#                 csu.tryLogin()
#
#     return csu.logged_in()


if not csu.logged_in():
    placeholder = st.empty()
    succeed_to_login = False
    with placeholder.form(key="login-form"):
        #login(placeholder)
        username = st.text_input('Username', value="paduraru2009").lower()
        password = st.text_input('Password', type='password', value="Arbori2009")

        if st.form_submit_button('Login'):
            if csu.tryLogin(username, password):
                placeholder.empty()
                succeed_to_login = True
            else:
                st.markdown(f'### Provided credentials are not correct')

    if succeed_to_login:
        st.markdown(f'### You are logged in!')
        csu.showLoggedUserSidebar()
        st.rerun()

else:
    st.markdown(f'You are logged in!')
    csu.showLoggedUserSidebar()