from loguru import logger
import time, sys
from typing import List


class Logger:
    _LOG_LEVEL = "ERROR"
    _LOG_LEVELS = [
        "CRITICAL",
        "ERROR",
        "WARNING",
        "INFO",
        "DEBUG",
        "TRACE",
    ]
    _LOGGER_NAME = "ASR4"
    _FILTERS = {"numba": "INFO", "asyncio": "WARNING", "grpc": "WARNING"}

    def __init__(self, logLevel: str = _LOG_LEVEL) -> None:
        logLevel = self.__validateLogLevel(logLevel)
        logger.info(f"Logging Level set to '{logLevel}'")
        self.__configureLogger(logLevel)

    def __validateLogLevel(self, logLevel: str) -> int:
        if logLevel not in Logger._LOG_LEVELS:
            logger.error(
                f"Level '{logLevel}' is not valid log level. Will use '{Logger.getDefaultLevel()}' instead."
            )
            logLevel = Logger.getDefaultLevel()
        return logLevel

    def __configureLogger(self, logLevel: str) -> None:
        logger.remove()
        logger.configure(extra={"user_id": "unknown", "transcription_id": "unknown"})
        logger.add(
            sys.stdout,
            level=logLevel,
            format="[{time:YYYY-MM-DDTHH:mm:ss.SSS}Z <level>{level}</level> <magenta>{module}</magenta>::<magenta>{function}</magenta>]"
            "[{extra[user_id]}][{extra[transcription_id]}] "
            "<level>{message}</level>",
            enqueue=True,
            filter=self._FILTERS,
        )

    @staticmethod
    def getLevels() -> List[str]:
        return Logger._LOG_LEVELS

    @staticmethod
    def getDefaultLevel() -> str:
        return Logger._LOG_LEVEL
