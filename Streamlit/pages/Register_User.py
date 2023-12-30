import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from userUtils import SecurityOfficer, SecurityOfficerExpertise, ResonsePreferences, Preference_Politely, Preference_Emojis
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
        st.session_state['InProgRegistration'] = SecurityOfficer()

    return st.session_state['REG_STATE']

def getInProgressRegistrationUser()-> SecurityOfficer:
    return st.session_state['InProgRegistration']

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
                    succeed = True
                    userInProgress = getInProgressRegistrationUser()
                    userInProgress.username = new_username
                    userInProgress.email = new_email
                    userInProgress.birthday = birthday
                    userInProgress.name = new_name
                    userInProgress.password = new_password
                    userInProgress.picture = picture
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

    csu.ShowTODO("TODO SSH: Will tune this set of questions")

    user = getInProgressRegistrationUser()

    # How likely is to attack from inside
    internalDamageChoice_options = [":rainbow[No, ever]", "Just for fun or to test my thing :movie_camera:", "***Maybe***"]
    internalDamageChoice = register_user_form.radio(
    "Would you ever try to test our production systems reliability from your own perspective without having your "
     "lead to know about it ? (Hidden: this is the intentional damage factor thing)",
        options=internalDamageChoice_options, index =0,
    )
    internalDamageChoice = internalDamageChoice_options.index(internalDamageChoice)
    if internalDamageChoice == 0:
        user.intentional_damage_factor = 0.0
    elif internalDamageChoice == 1:
        user.intentional_damage_factor = 0.5
    else:
        user.intentional_damage_factor = 1.0

    user.correct_teamwork = register_user_form.slider("How likely is to report a collegue that brings in without authorization own software, "
    "or causes intentional damage ? (Hidden: Correct teamwork factor)",
              min_value=1, max_value=10, value=5, step=1) / 10.0

    motivationValue = register_user_form.slider("How motivated are you with what you do daily"
                                      " (Hidden : Motivation factor) ?",
              min_value=1, max_value=10, value=5, step=1) / 10.0

    confidence_options = ['Very confident', 'Somehow confident', 'No confidence at all']
    confidenceChoise = register_user_form.selectbox("How confident ar you on your job  (Hidden : Motivation factor) ?",
        options=confidence_options)

    confidenceChoise = confidence_options.index(confidenceChoise)
    confidenceValue = 1.0
    if confidenceChoise == 1:
        confidenceValue = 0.5
    elif confidenceChoise == 2:
        confidenceValue = 1.0

    user.motivation_factor = (motivationValue + confidenceValue) / 2.0

    if register_user_form.form_submit_button('Next'):
        succeed = True

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
    csu.ShowTODO("TODO FMI: this will be an LLM guided interview with questions based on your uploaded CV,"
            "previous answers and feedback. A dynamic interview with different threads of discussions. Stay tuned!")

    user = getInProgressRegistrationUser()


    # Expertise evaluation
    experience_options = ["1-3 years", "3-7 years", ">7 years"]
    experienceChoice = register_user_form.radio("How many years of expertise do you have in security mechanisms?",
                                                options=experience_options, index=0)
    experienceChoice = experience_options.index(experienceChoice)
    user.expertise = SecurityOfficerExpertise.BEGINNER
    if experienceChoice == 1:
        user.expertise = SecurityOfficerExpertise.MIDDLE
    else:
        user.expertise = SecurityOfficerExpertise.ADVANCED

    fulled_externaldevices_options = ['I do not know', 'No', 'Maybe']
    fulled_phishing_choice = register_user_form.selectbox("Would you open an external link or program in "
                                                    "an email from colleagues, friends, family ? "
                                                    "(Hidden: how easy is to be fulled)",
                                                    options=fulled_externaldevices_options)

    fulled_phishing_choice = fulled_externaldevices_options.index(fulled_phishing_choice)
    fulled_phishing_value = 1.0
    if fulled_phishing_choice == 1:
        fulled_phishing_value = 0.5
    elif fulled_phishing_choice == 2:
        fulled_phishing_value = 1.0

    fulled_externaldevices_options = ['I do not know', 'No', 'Maybe']
    fulled_externaldevices_choice = register_user_form.selectbox("Would you bring a USB stick to company's provided "
                                                          "for your personal things or being asked by a colleague?",
                                                            options=fulled_externaldevices_options)

    fulled_externaldevices_choice = fulled_externaldevices_options.index(fulled_externaldevices_choice)
    fulled_externaldevices_value = 1.0
    if fulled_externaldevices_choice == 1:
        fulled_externaldevices_value = 0.5
    elif fulled_externaldevices_choice == 2:
        fulled_externaldevices_value = 1.0

    user.can_be_tricked_out_factor = (fulled_externaldevices_value + fulled_phishing_value) * 0.5

    if register_user_form.form_submit_button('Next'):
        succeed = True

    return succeed

def register_user_preferences(form_name: str, location: str = 'main') -> bool:
    """
    Creates a register new user widget.
    """
    csu.ShowTODO("TODO SSH: Will tune this set of questions")

    succeed = False

    if location not in ['main', 'sidebar']:
        raise ValueError("Location must be one of 'main' or 'sidebar'")
    if location == 'main':
        register_user_form = st.form('User preferences')
    elif location == 'sidebar':
        register_user_form = st.sidebar.form('User preferences')

    register_user_form.subheader(form_name)
    user = getInProgressRegistrationUser()

    user.preference = ResonsePreferences.DETAILED \
        if register_user_form.toggle('Use detailed answers instead of concise', True) is True \
        else ResonsePreferences.CONCISE

    user.emojis = Preference_Emojis.USE_EMOJIS \
        if register_user_form.toggle('Use emojis in answers', True) is True \
        else Preference_Emojis.NO_EMOJIS

    user.politely = Preference_Politely.POLITE_PRESENTATION \
        if register_user_form.toggle('Prefer chatbot polite presentations', True) is True \
        else Preference_Politely.FORMAL_PRESENTATION

    if register_user_form.form_submit_button('Next'):
        succeed = True

    return succeed


if csu.logged_in():
    st.write("## Already logged in, logout please to continue!")
    csu.showLoggedUserSidebar()
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
        res = register_user_technical("Technical registration")
        if res:
            updateRegistrationState(RegistrationState.PREFERENCES_REGISTRATION)
            st.rerun()
    elif getRegistrationState() == RegistrationState.PREFERENCES_REGISTRATION:
        res = register_user_preferences("Preferences registration")
        if res:
            updateRegistrationState(None)
            csu.register_credentials(getInProgressRegistrationUser())
            switch_page("Main Page")
            st.rerun()

