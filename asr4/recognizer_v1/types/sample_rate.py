from enum import Enum, unique
from typing import Optional
from typing_extensions import Self


@unique
class SampleRate(Enum):
    HZ_16000 = 16000

    @classmethod
    def parse(cls, sample_rate_hz: int) -> Optional[Self]:
        for rate in cls:
            if rate.value == sample_rate_hz:
                return rate
        return None

    @classmethod
    def check(cls, sample_rate_hz: int) -> bool:
        return cls.parse(sample_rate_hz) is not None
