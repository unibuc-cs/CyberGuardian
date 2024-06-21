import datetime
from enum import IntEnum
from typing import Dict, List, Union


class SecurityOfficerExpertise(IntEnum):
    BEGINNER = 0
    MIDDLE = 1
    ADVANCED = 2


def getUserExpertiseStr(exp: int) -> str:
    if exp == SecurityOfficerExpertise.BEGINNER:
        return "Beginner"
    elif exp == SecurityOfficerExpertise.MIDDLE:
        return "MIDDLE"
    elif exp == SecurityOfficerExpertise.ADVANCED:
        return "ADVANCED"
    else:
        assert False, "Unexpected security officer"


class Response_Preferences(IntEnum):
    DETAILED = 0
    CONCISE = 1


class Preference_Politely(IntEnum):
    POLITE_PRESENTATION = 0
    FRIENDLY_PRESENTATION = 1


class Preference_Emojis(IntEnum):
    USE_EMOJIS = 0
    NO_EMOJIS = 1


class Preference_FeedbackArea(IntEnum):
    ALLOW_FEEDBACK_ON_HISTORY = 0
    ALLOW_FEEDBACK_ONLY_ON_LAST_MSG = 1
class ACL_Role(IntEnum):
    ADMIN = 0
    USER = 1

class ACL_Permission(IntEnum):
    NONE = 0
    # Can read the data
    READ = 1
    # Can write the data
    WRITE = 2
    # Can delete the data
    DELETE = 3
    # Can execute code on the data
    EXECUTE = 4




class SecurityOfficer:
    def __init__(self):
        super().__init__()

        self.logged_in: bool = False

        self.name: str = "defaultUser"
        self.username: str = "defaultUserName"
        self.password: str = "defaultUserPassword"
        self.birthday: datetime.datetime = datetime.datetime.now()

        self.expertise: SecurityOfficerExpertise = SecurityOfficerExpertise.ADVANCED
        self.preference: Response_Preferences = Response_Preferences.CONCISE
        self.politely: Preference_Politely = Preference_Politely.FRIENDLY_PRESENTATION
        self.emojis: Preference_Emojis = Preference_Emojis.NO_EMOJIS
        self.feedbackArea: Preference_FeedbackArea = Preference_FeedbackArea.ALLOW_FEEDBACK_ONLY_ON_LAST_MSG

        self.motivation_factor: float = 1.0  # A float between 0-1

        # Some technical evaluate question
        self.technical_responses_eval: float = 1.0

        # How easy it can be tricked out to give access to hackers
        # E.g. phishing attack, give password, stick etc.
        # 0-1
        self.can_be_tricked_out_factor: float = 0.0

        # The teamwork - technical way
        self.team_work_tech_value: float = 0.0

        # How likely is to attack from inside
        # 0-1
        self.intentional_damage_factor: float = 0.0

        # If 0 - does not report when seeing strange situation from colleagues, 1 - report everything as expected
        self.correct_teamwork: float = 1.0

        self.avatar_choice: str = "Data/characters/woman.png"

        # TODO should assign this from the database or by the user profile evaluation.

        self.AC_Role: ACL_Role = ACL_Role.USER

        #self.AC_Permission: ACL_Permission = ACL_Permission.READ
        self.AC_Permission: ACL_Permission = ACL_Permission.EXECUTE

    def __repr__(self):
        return "".join({f"{key}:{value}\n" for key, value in self.__dict__.items()})

    """
    def login(self, user_settings: Dict):
        for key, value in user_settings.items():
            setattr(self, key, value)
        self.loggedin = True
    """
