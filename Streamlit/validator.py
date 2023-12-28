import re
import datetime
from dateutil.relativedelta import relativedelta

class validator:
    @staticmethod
    def validate_username(username: str) -> bool:
        pattern = r"^[a-zA-Z0-9_-]{1,20}$"
        return bool(re.match(pattern, username))
    @staticmethod
    def validate_name(name: str) -> bool:
        return 1 < len(name) < 100


    @staticmethod
    def validate_email(email: str) -> bool:
        return "@" in email and 2 < len(email) < 320

    @staticmethod
    def validate_birthday(birthday: datetime.datetime) -> bool:
        difference_in_years = relativedelta(datetime.datetime.now(), birthday).years
        return difference_in_years >= 18
