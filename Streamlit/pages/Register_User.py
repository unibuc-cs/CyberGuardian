import streamlit as st
from userUtils import SecurityOfficer
import clientserverUtils as csu
from validator import validator
import datetime
from enum import IntEnum
from databaseUtils import CredentialsDB
from typing import Union

class RegistrationState(IntEnum):
    BASIC_REGISTRATION = 0
    BEHAVIORAL_REGISTRATION = 1
    TECHNICAL_REGISTRATION = 2
    PREFERENCES_REGISTRATION = 3
    FINISHED_REGISTRATION = 4

if 'cdb' not in st.session_state:
    csu.checkCreateCredentialsDB()

def getRegistrationState() -> RegistrationState:
    res = st.session_state.get('REG_STATE', None)
    if res is None:
        st.session_state['REG_STATE'] = RegistrationState.BASIC_REGISTRATION

    return st.session_state['REG_STATE']

def updateRegistrationState(newRegState: Union[RegistrationState, None]):
    st.session_state['REG_STATE'] = newRegState

def register_user_basic(form_name: str, location: str = 'main') -> bool:
    """
    Creates a register new user widget.
    """
    succeed = False

    if location not in ['main', 'sidebar']:
        raise ValueError("Location must be one of 'main' or 'sidebar'")
    if location == 'main':
        register_user_form = st.form('Basic registration')
    elif location == 'sidebar':
        register_user_form = st.sidebar.form('Basic registration')

    register_user_form.subheader(form_name)
    new_email = register_user_form.text_input('Email', value="ciprian.paduraru2009@gmail.com")
    new_username = register_user_form.text_input('Username', value="paduraru2009").lower()
    new_name = register_user_form.text_input('Name', value="Ciprian Paduraru")
    new_password = register_user_form.text_input('Password', type='password', value="Arbori2009")
    new_password_repeat = register_user_form.text_input('Repeat password', type='password', value="Arbori2009")
    birthday = register_user_form.date_input("When's your birthday",
                                             value=datetime.date(1986, 9, 3),
                                             max_value=datetime.date.today(),
                                             min_value=datetime.date(1900, 7, 6))

    picture = register_user_form.camera_input("Take a picture of yourself")

    if register_user_form.form_submit_button('Next'):
        if len(new_email) > 0 and len(new_username) > 0 and len(new_name) > 0 and len(new_password) > 0 and (picture is not None):
            validSetup = True
            if not validator.validate_username(new_username):
                csu.RegisterError("Invalid username")
                validSetup = False
            if not validator.validate_email(new_email):
                csu.RegisterError("Invalid email")
                validSetup = False
            if not validator.validate_name(new_name):
                csu.RegisterError("Invalid name")
                validSetup = False
            if not validator.validate_birthday(birthday):
                validSetup = False
                csu.RegisterError("Invalid birthday")
            if not picture:
                csu.RegisterError("No picture provided")
                validSetup = False
            if new_password != new_password_repeat:
                csu.RegisterError("Passwords do not match")

            if validSetup:
                if csu.isValidNewUsername(new_username):
                    csu.register_credentials(new_username, new_name, new_password, new_email, birthday, picture)
                    succeed = True
                else:
                    csu.RegisterError('Username already taken')
            else:
                pass
                #csu.RegisterError('Username already taken')
        else:
            csu.RegisterError('Please fill in data and take the photo')

    return succeed

def register_user_behavioral(form_name: str, location: str = 'main') -> bool:
    """
    Creates a register new user widget.
    """
    succeed = False

    if location not in ['main', 'sidebar']:
        raise ValueError("Location must be one of 'main' or 'sidebar'")
    if location == 'main':
        register_user_form = st.form('Behavioral profile')
    elif location == 'sidebar':
        register_user_form = st.sidebar.form('Behavioral profile')

    register_user_form.subheader(form_name)
    register_user_form.write("TODO")
    if register_user_form.form_submit_button('Next'):
        succeed = True
        # TODO: update user in db
        pass

    return succeed

def register_user_technical(form_name: str, location: str = 'main') -> bool:
    """
    Creates a register new user widget.
    """
    succeed = False

    if location not in ['main', 'sidebar']:
        raise ValueError("Location must be one of 'main' or 'sidebar'")
    if location == 'main':
        register_user_form = st.form('Technical evaluation')
    elif location == 'sidebar':
        register_user_form = st.sidebar.form('Technical evaluation')

    register_user_form.subheader(form_name)
    register_user_form.write("TODO")
    if register_user_form.form_submit_button('Next'):
        succeed = True
        # TODO: update user in db
        pass

    return succeed

def register_user_preferences(form_name: str, location: str = 'main') -> bool:
    """
    Creates a register new user widget.
    """
    succeed = False

    if location not in ['main', 'sidebar']:
        raise ValueError("Location must be one of 'main' or 'sidebar'")
    if location == 'main':
        register_user_form = st.form('User preferences')
    elif location == 'sidebar':
        register_user_form = st.sidebar.form('User preferences')

    register_user_form.subheader(form_name)
    register_user_form.write("TODO")
    if register_user_form.form_submit_button('Next'):
        succeed = True
        # TODO: update user in db
        pass

    return succeed


if csu.logged_in():
    st.write("## Already logged in, logout please to continue!")
else:
    if getRegistrationState() == RegistrationState.BASIC_REGISTRATION:
        res = register_user_basic("Basic registration")
        if res:
            updateRegistrationState(RegistrationState.BEHAVIORAL_REGISTRATION)
            st.rerun()
    elif getRegistrationState() == RegistrationState.BEHAVIORAL_REGISTRATION:
        res = register_user_behavioral("Behavioral registration")
        if res:
            updateRegistrationState(RegistrationState.TECHNICAL_REGISTRATION)
            st.rerun()
    elif getRegistrationState() == RegistrationState.TECHNICAL_REGISTRATION:
        res = register_user_behavioral("Technical registration")
        if res:
            updateRegistrationState(RegistrationState.PREFERENCES_REGISTRATION)
            st.rerun()
    elif getRegistrationState() == RegistrationState.PREFERENCES_REGISTRATION:
        res = register_user_behavioral("Preferences")
        if res:
            updateRegistrationState(None)
            st.rerun()

