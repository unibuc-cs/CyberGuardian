
import streamlit as st
from userUtils import SecurityOfficer
import clientserverUtils as csu

def check_cookie():
    pass
def login(self, form_name: str, location: str = 'main') -> tuple:
    """
    Creates a login widget.

    Parameters
    ----------
    form_name: str
        The rendered name of the login form.
    location: str
        The location of the login form i.e. main or sidebar.
    Returns
    -------
   Fills in the user in session state
    """
    if location not in ['main', 'sidebar']:
        raise ValueError("Location must be one of 'main' or 'sidebar'")
    if not csu.logged_in():
        check_cookie()
        if not csu.logged_in():
            if location == 'main':
                login_form = st.form('Login')
            elif location == 'sidebar':
                login_form = st.sidebar.form('Login')

            login_form.subheader(form_name)
            self.username = login_form.text_input('Username').lower()
            st.session_state['username'] = self.username
            self.password = login_form.text_input('Password', type='password')

            if login_form.form_submit_button('Login'):
                self._check_credentials()

    return st.session_state['name'], st.session_state['authentication_status'], st.session_state['username']