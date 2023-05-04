from pyformatter import PyFormatter as Formatter
from asr4.types.language import Language


class FormatterFactory:
    @staticmethod
    def createFormatter(model_path: str, language: Language) -> Formatter:
        return Formatter(
            FormatterFactory._sanitizeLanguage(language.asFormatter()),
            model_path,
            b"",
            b"",
            dict(),
        )

    def _sanitizeLanguage(language: str) -> str:
        if len(language) == 2:
            return language + "-" + language
        return language
