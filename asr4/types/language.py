from enum import Enum, unique
from typing import Optional
from typing_extensions import Self
from typing import Union


@unique
class Language(Enum):
    EN_US = "en-US"
    ES = "es"
    PT_BR = "pt-BR"

    @classmethod
    def parse(cls, language: Union[str, Self]) -> Optional[Self]:
        if isinstance(language, Language):
            return language
        try:
            return cls[language.upper().replace("-", "_")]
        except:
            return None

    @classmethod
    def check(cls, language: str) -> bool:
        return cls.parse(language) is not None

    def asFormatter(self) -> str:
        return self.value.lower()

    @staticmethod
    def getValidOptions():
        return [Language.EN_US, Language.ES, Language.PT_BR]

    @staticmethod
    def getDefaultValue():
        return Language.EN_US
