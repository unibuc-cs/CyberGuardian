from enum import Enum

class SecurityOfficerExpertise(Enum):
    BEGINNER = 0
    MIDDLE = 1
    ADVANCED = 2

class Preference_ResponseType(Enum):
    DETAILED = 0
    CONCISE = 1


class Preference_Politely(Enum):
    POLITE_PRESENTATION = 0
    FORMAL_PRESENTATION = 1

class Preference_Emojis(Enum):
    USE_EMOJIS = 1
    NO_EMOJIS = 2

class SecurityOfficer():
    def __init__(self):
        super().__init__()

        self.loggedin : bool = False

        self.name: str = ""
        self.username: str = ""
        self.password: str = ""

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

    def isLoggedIn(self) -> bool:
        return self.loggedin