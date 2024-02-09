from datetime import datetime
from typing import Optional


def get_current_datetime_str(no_micro: Optional[bool] = False) -> str:
    return datetime.now().strftime(
        "%m-%d:%H:%M:%S:%f" if not no_micro else "%m-%d:%H:%M:%S"
    )


def get_current_ms() -> str:
    return datetime.now().strftime("%f")
