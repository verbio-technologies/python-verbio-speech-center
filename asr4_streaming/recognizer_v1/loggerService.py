from loguru import logger
#import logging
import logging.handlers
import time
import multiprocessing
import traceback
import sys
from typing import List


#class Logger(logger):
    # def __init__(self, name: str, level: int, queue: multiprocessing.Queue):
    #     super().__init__(name, level)
    #     self.addHandler(logging.handlers.QueueHandler(queue))
    #     self.setLevel(level)

#    def __init__(self, name: str, level: str, queue: multiprocessing.Queue):
#        super().__init__(name, level)
#        self.add(queue, level=level, enqueue=True)

def filterMyMessages(record):
    print("[+]",record)
    return record["level"].name == "INFO"

class LoggerQueue:
    def __init__(
        self, logger_name: str, log_level: str, logsQueue: multiprocessing.Queue
    ):
        self._logger_name = logger_name
        self._log_level = log_level
        self._queue = logsQueue

    def getLogger(self):
        logger.add(logging.handlers.QueueHandler(self._queue),
                   level=self._log_level,
                   filter=filterMyMessages,
                   enqueue=True)
        return logger
        # return Logger(self._logger_name, self._log_level, self._queue)

    def configureGlobalLogger(self):
        LoggerService.configureLogger(self._log_level)


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

    def __init__(
        self, log_level: str = _LOG_LEVEL, logger_name: str = _LOGGER_NAME
    ) -> None:
        self._stopSignal = multiprocessing.Event()
        self._logger_name = logger_name
        self._log_level = self.validateLogLevel(log_level)
        self._queue = multiprocessing.Queue(-1)
        self.logger = self.getQueue().getLogger()
        self.logger.log(self._log_level, "Loglevel set to: " + self._log_level)
        self._spawnService()

    def getQueue(self):
        return LoggerQueue(self._logger_name, self._log_level, self._queue)

    def getLogger(self):
        return self.getQueue().getLogger()

    def stop(self):
        self._stopSignal.set()
        self._listener.join(5)
        self._listener.kill()

    def configureGlobalLogger(self):
        LoggerService.configureLogger(self._log_level)

    @staticmethod
    def getLogLevelOptions() -> List[str]:
        return list(LoggerService._LOG_LEVELS.keys())

    @staticmethod
    def getDefaultLogLevel() -> str:
        return LoggerService._LOG_LEVEL

    @staticmethod
    def configureLogger(level: int) -> None:
        filters = {"numba": "INFO", "asyncio": "WARNING", "grpc": "WARNING"}
        logger.add(sys.stdout, filter=filters, format="[{time:YYYY-MM-DDTHH:mm:ss.SSS}Z <level>{level}</level> <magenta>{module}</magenta>::<magenta>{function}</magenta>]"
                   # "[{extra[user_id]}][{extra[transcription_id]}] "
                   "<level>{message}</level>",
                   enqueue=True)
        #logging.Formatter.converter = time.gmtime
        #logging.basicConfig(
        #    level=level,
        #    format="[%(asctime)s.%(msecs)03d %(levelname)s %(name)s::%(module)s::%(funcName)s] (PID %(process)d): %(message)s",
        #    datefmt="%Y-%m-%d %H:%M:%S",
        #)

    def _spawnService(self):
        self._stopSignal = multiprocessing.Event()
        self._listener = multiprocessing.Process(
            target=LoggerService._server_logger_listener,
            args=(self._queue, self._log_level, self._stopSignal),
        )
        self._listener.start()

    @staticmethod
    def _server_logger_listener(
        queue: multiprocessing.Queue, level: int, stopSignal: multiprocessing.Event()
    ):
        LoggerService.configureLogger(level)
        while not stopSignal.is_set() or not queue.empty():
            try:
                record = queue.get()
                if record is None:
                    break
                logger = logging.getLogger(record.name)
                logger.handle(record)
            except Exception:
                print("[ERROR] Couldn't write log", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    def validateLogLevel(self, loglevel: str) -> int:
        if loglevel not in LoggerService._LOG_LEVELS:
            offender = loglevel
            loglevel = LoggerService._LOG_LEVEL
            logger.error(f"Level {offender} is not valid log level. Will use {loglevel} instead.")
        return loglevel
