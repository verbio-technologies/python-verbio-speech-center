from pyformatter import PyFormatter as Formatter
from asr4.types.language import Language


class FormatterFactory:
    @staticmethod
    def createFormatter(model_path: str, language: Language) -> Formatter:
        return Formatter(language.asFormatter(), model_path, b"", b"", dict())
