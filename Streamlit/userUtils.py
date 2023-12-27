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

        self.expertise : SecurityOfficerExpertise = SecurityOfficerExpertise.BEGINNER
        self.preference : Preference_ResponseType = Preference_ResponseType.DETAILED
        self.politely : Preference_Politely = Preference_Politely.POLITE_PRESENTATION
        self.emojis : Preference_Emojis = Preference_Emojis.USE_EMOJIS

