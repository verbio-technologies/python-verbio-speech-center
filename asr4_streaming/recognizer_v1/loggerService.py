from loguru import logger
import time, traceback, sys
from typing import List


class LoggerService:
    _LOG_LEVELS = {
        "ERROR": logger.level("ERROR").no,
        "WARNING": logger.level("WARNING").no,
        "WARN": logger.level("WARNING").no,
        "INFO": logger.level("INFO").no,
        "DEBUG": logger.level("DEBUG").no,
        "TRACE": logger.level("DEBUG").no,
    }
    _LOG_LEVEL = "ERROR"
    _LOGGER_NAME = "ASR4"

    def __init__(self, level: int) -> None:
        self._configure(level)

    @staticmethod
    def getLogLevelOptions() -> List[str]:
        return list(LoggerService._LOG_LEVELS.keys())

    @staticmethod
    def getDefaultLogLevel() -> str:
        return LoggerService._LOG_LEVEL

    def _configure(self, level:int) -> None:
        logger.remove()
        logger.configure(extra={"user_id": "unknown", "transcription_id": "unknown"})
        log_level = self.validateLogLevel(level)
        logger.log(log_level, "Loglevel set to: " + log_level)
        filters = {"numba": "INFO", "asyncio": "WARNING", "grpc": "WARNING"}
        logger.add(
            sys.stdout,
            filter=filters,
            format="[{time:YYYY-MM-DDTHH:mm:ss.SSS}Z <level>{level}</level> <magenta>{module}</magenta>::<magenta>{function}</magenta>]"
            "[{extra[user_id]}][{extra[transcription_id]}] "
            "<level>{message}</level>",
            enqueue=True,
        )
        # logging.basicConfig(
        #    level=level,
        #    format="[%(asctime)s.%(msecs)03d %(levelname)s %(name)s::%(module)s::%(funcName)s] (PID %(process)d): %(message)s",
        #    datefmt="%Y-%m-%d %H:%M:%S",
        # )

    def validateLogLevel(self, loglevel: str) -> int:
        if loglevel not in LoggerService._LOG_LEVELS:
            offender = loglevel
            loglevel = LoggerService._LOG_LEVEL
            logger.error(
                f"Level {offender} is not valid log level. Will use {loglevel} instead."
            )
        return loglevel
