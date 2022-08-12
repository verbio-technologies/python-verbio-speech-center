from enum import Enum, unique
from typing import Optional
from typing_extensions import Self

@unique
class Language(Enum):
    EN_US = 'en-US'
    ES_ES = 'es-ES'
    PT_BR = 'pt-BR'

    @classmethod
    def parse(
        cls, 
        language: str
    ) -> Optional[Self]:
        try:
            return cls[language.upper().replace('-', '_')]
        except:
            return None

    @classmethod
    def check(
        cls, 
        language: str
    ) -> bool:
        return cls.parse(language) is not None    