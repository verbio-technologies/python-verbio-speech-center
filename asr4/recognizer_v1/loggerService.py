import logging
import logging.handlers
import time
import multiprocessing
import traceback
import sys


class Logger(logging.Logger):
    def __init__(self, name: str, level: int, queue: multiprocessing.Queue):
        super().__init__(name, level)
        self.addHandler(logging.handlers.QueueHandler(queue))
        self.setLevel(level)


class LoggerQueue:
    def __init__(
        self, logger_name: str, log_level: int, logsQueue: multiprocessing.Queue
    ):
        self._logger_name = logger_name
        self._log_level = log_level
        self._queue = logsQueue

    def getLogger(self):
        return Logger(self._logger_name, self._log_level, self._queue)

    def configureGlobalLogger(self):
        LoggerService.configureLogger(self._log_level)


class LoggerService:
    _LOG_LEVELS = {
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "TRACE": logging.DEBUG,
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
        self.logger.log(
            self._log_level, "Loglevel set to: " + logging.getLevelName(self._log_level)
        )
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
    def getLogLevelOptions() -> []:
        return list(LoggerService._LOG_LEVELS.keys())

    @staticmethod
    def getDefaultLogLevel() -> str:
        return LoggerService._LOG_LEVEL

    @staticmethod
    def configureLogger(level: int) -> None:
        logging.getLogger("numba").setLevel(min(level, logging.INFO))
        logging.Formatter.converter = time.gmtime
        logging.basicConfig(
            level=level,
            format="[%(asctime)s.%(msecs)03d %(levelname)s %(name)s::%(module)s::%(funcName)s] (PID %(process)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

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
            self.logger.error(
                "Level [%s] is not valid log level. Will use %s instead."
                % (offender, loglevel)
            )
        return self._LOG_LEVELS.get(loglevel, self._LOG_LEVEL)
