from enum import Enum, unique
from typing import Optional
from typing_extensions import Self


@unique
class AudioEncoding(Enum):
    PCM = 0

    @classmethod
    def parse(cls, audio_encoding: int) -> Optional[Self]:
        for encoding in cls:
            if encoding.value == audio_encoding:
                return encoding
        return None

    @classmethod
    def check(cls, audio_encoding: int) -> bool:
        return cls.parse(audio_encoding) is not None
