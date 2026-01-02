import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger()

ID = str(int(time.time()))


@dataclass
class InputDate:
    """
    DataClass to store different formats of date
    """

    date: str
    time_delta_days: int

    def __post_init__(self):
        self.to_date_str = self.date
        self.to_date_dt = datetime.strptime(self.to_date_str, "%Y-%m-%d")
        self.to_date_no_hyph = self.to_date_str.replace("-", "")
        self.from_date_dt = self.to_date_dt - timedelta(days=self.time_delta_days)
        self.from_date_str = self.from_date_dt.strftime("%Y-%m-%d")
        self.from_date_no_hyph = self.from_date_dt.strftime("%Y%m%d")


def apply_config_to_data(data, attribute, params):
    if attribute in params.keys():
        setattr(data, attribute, params[attribute])
