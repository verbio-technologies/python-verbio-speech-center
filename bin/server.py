import sys, traceback

import grpc
import asyncio
import logging
import logging.handlers
import argparse
import multiprocessing
import time
from concurrent import futures

from typing import Optional

from asr4.recognizer import Language
from asr4.recognizer import SERVICES_NAMES
from asr4.recognizer import OnnxSession
from asr4.recognizer import RecognizerService
from asr4.recognizer import add_RecognizerServicer_to_server

from grpc_health.v1 import health
from grpc_health.v1.health_pb2 import HealthCheckResponse
from grpc_health.v1.health_pb2_grpc import add_HealthServicer_to_server


_PROCESS_COUNT = multiprocessing.cpu_count()
_LOG_LEVELS = {1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
_LOG_LEVEL = 2


def server_logger_listener(queue, verbose):
    server_logger_configurer(verbose)
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            print("Couldn't write log", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def server_logger_configurer(level):
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=_LOG_LEVELS.get(level, logging.INFO),
        format="[%(asctime)s.%(msecs)03d %(levelname)s %(module)s::%(funcName)s] (PID %(process)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def serve(
    args: argparse.Namespace,
) -> None:
    queue = multiprocessing.Queue(-1)
    loglevel = args.verbose
    listener = multiprocessing.Process(
        target=server_logger_listener, args=(queue, loglevel)
    )
    listener.start()
    _log_server_configurer(queue, loglevel)
    logger = logging.getLogger("ASR4")
    logger.info("Binding to '%s'", args.bindAddress)

    workers = []
    providers = ["CPUExecutionProvider"]
    if args.gpu:
        providers = ["CUDAExecutionProvider"] + providers
    for _ in range(args.jobs):
        worker = multiprocessing.Process(
            target=_asyncRunServer,
            args=(
                queue,
                args.bindAddress,
                args.model,
                providers,
                Language.parse(args.language),
                args.vocabulary,
                args.formatter,
                args.jobs,
                loglevel,
            ),
        )
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()


def _asyncRunServer(
    queue: multiprocessing.Queue,
    bindAddress: str,
    model: str,
    providers: str,
    language: Language,
    vocabulary: Optional[str],
    formatter: Optional[str],
    jobs: int,
    loglevel: int,
) -> None:
    asyncio.run(
        _runServer(
            queue,
            bindAddress,
            model,
            providers,
            language,
            vocabulary,
            formatter,
            jobs,
            loglevel,
        )
    )


def _log_server_configurer(queue, level):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger("ASR4")
    root.addHandler(h)
    root.setLevel(_LOG_LEVELS.get(level, logging.INFO))


async def _runServer(
    queue: multiprocessing.Queue,
    bindAddress: str,
    model: str,
    providers: str,
    language: Language,
    vocabularyPath: Optional[str],
    formatterPath: Optional[str],
    jobs: int,
    loglevel: int,
) -> None:
    _log_server_configurer(queue, loglevel)
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=jobs),
        options=(("grpc.so_reuseport", 1),),
    )
    _addRecognizerService(
        server, model, providers, language, vocabularyPath, formatterPath
    )
    _addHealthCheckService(server, jobs)
    server.add_insecure_port(bindAddress)
    logger = logging.getLogger("ASR4")
    logger.info(f"Server listening {bindAddress}")
    await server.start()
    await server.wait_for_termination()


def _addRecognizerService(
    server: grpc.aio.Server,
    model: str,
    providers: str,
    language: Language,
    vocabularyPath: Optional[str],
    formatterPath: Optional[str],
) -> None:
    session = OnnxSession(
        model,
        providers=providers,
    )
    add_RecognizerServicer_to_server(
        RecognizerService(session, language, vocabularyPath, formatterPath), server
    )


def _addHealthCheckService(
    server: grpc.aio.Server,
    jobs: int,
) -> None:
    healthServicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=jobs),
    )
    _markAllServicesAsHealthy(healthServicer)
    add_HealthServicer_to_server(healthServicer, server)


def _markAllServicesAsHealthy(healthServicer: health.HealthServicer) -> None:
    for service in SERVICES_NAMES + [health.SERVICE_NAME]:
        healthServicer.set(service, HealthCheckResponse.SERVING)


def _parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python ASR4 Server")
    parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        dest="model",
        help="Path to the model file.",
    )
    parser.add_argument(
        "-d",
        "--dictionary-path",
        dest="vocabulary",
        help="Path to the model's dictionary file, containing all the possible outputs from the model.",
    )
    parser.add_argument(
        "-l",
        "--language",
        dest="language",
        default=Language.EN_US.value,
        choices=[l.value.lower() for l in Language],
        type=str.lower,
        help="Language of the recognizer service.",
    )
    parser.add_argument(
        "-f",
        "--formatter-model-path",
        dest="formatter",
        help="Path to the formatter model file.",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        dest="gpu",
        default=False,
        action="store_true",
        help="Whether to use GPU instead of CPU",
    )
    parser.add_argument(
        "--host",
        dest="bindAddress",
        default="[::]:50051",
        help="Hostname address to bind the server to.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        dest="jobs",
        default=_PROCESS_COUNT,
        help="Number of parallel workers; if not specified, defaults to CPU count.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=_LOG_LEVEL,
        help="Give more output. Option is additive, and can be used up to 2 times to achieve INFO and DEBUG levels. Default level is WARNING.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = _parseArguments()
    if not Language.check(args.language):
        raise ValueError(f"Invalid language '{args.language}'")
    if args.verbose > 4:
        args.verbose = 4
    serve(args)
