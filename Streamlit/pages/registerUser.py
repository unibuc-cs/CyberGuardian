import streamlit as st
from userUtils import SecurityOfficer
import clientserverUtils as csu

def register_user(self, form_name: str, location: str = 'main', preauthorization=True) -> bool:
    """
    Creates a register new user widget.

    Parameters
    ----------
    form_name: str
        The rendered name of the register new user form.
    location: str
        The location of the register new user form i.e. main or sidebar.
    Returns
    -------
    bool
        The status of registering the new user, True: user registered successfully.
    """

    if location not in ['main', 'sidebar']:
        raise ValueError("Location must be one of 'main' or 'sidebar'")
    if location == 'main':
        register_user_form = st.form('Register user')
    elif location == 'sidebar':
        register_user_form = st.sidebar.form('Register user')

    register_user_form.subheader(form_name)
    new_email = register_user_form.text_input('Email')
    new_username = register_user_form.text_input('Username').lower()
    new_name = register_user_form.text_input('Name')
    new_password = register_user_form.text_input('Password', type='password')
    new_password_repeat = register_user_form.text_input('Repeat password', type='password')

    if register_user_form.form_submit_button('Register'):
        if len(new_email) and len(new_username) and len(new_name) and len(new_password) > 0:
            if new_username not in self.credentials['usernames']:
                if new_password == new_password_repeat:
                    if preauthorization:
                        if new_email in self.preauthorized['emails']:
                            self._register_credentials(new_username, new_name, new_password, new_email,
                                                       preauthorization)
                            return True
                        else:
                            raise csu.RegisterError('User not preauthorized to register')
                    else:
                        self._register_credentials(new_username, new_name, new_password, new_email, preauthorization)
                        return True
                else:
                    raise csu.RegisterError('Passwords do not match')
            else:
                raise csu.RegisterError('Username already taken')
        else:
            raise csu.RegisterError('Please enter an email, username, name, and password')