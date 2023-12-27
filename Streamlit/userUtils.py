from enum import Enum

class SecurityOfficerExpertise(Enum):
    BEGINNER = 1
    MIDDLE = 2
    ADVANCED = 3

class Preference_ResponseType(Enum):
    DETAILED = 1
    CONCISE = 2


class Preference_Politely(Enum):
    POLITE_PRESENTATION = 1
    FORMAL_PRESENTATION = 2

class Preference_Emojis(Enum):
    USE_EMOJIS = 1
    NO_EMOJIS = 2

class User:
    def __init__(self):
        self.__name : str= ""
        self.__loggedin : bool = False

    def isLoggedIn(self) -> bool:
        return self.__loggedin

    def getName(self) -> str:
        return self.__name

class SecurityOfficer(User):
    def __init__(self):
        super().__init__()

        self.expertise: SecurityOfficerExpertise = SecurityOfficerExpertise.BEGINNER
        self.preference: Preference_ResponseType = Preference_ResponseType.DETAILED
        self.politely: Preference_Politely = Preference_Politely.POLITE_PRESENTATION
        self.emojis: Preference_Emojis = Preference_Emojis.USE_EMOJIS

        self.motivation_factor: float = 1.0 # A float between 0-1

        # How easy it can be tricked out to give access to hackers
        # E.g. phishing attack, give password, stick etc.
        # 0-1
        self.can_be_tricked_out_factor: float = 0.0

        # How likely is to attack from inside
        # 0-1
        self.intentional_damage_factor: float = 0.0

        # If 0 - does not report when seeing strange situation from colleagues, 1 - report everything as expected
        self.correct_teamwork: float = 1.0