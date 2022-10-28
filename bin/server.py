import grpc
import asyncio
import logging
import argparse
import multiprocessing
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
_LOGGER = logging.getLogger(__name__)
_LOG_LEVELS = {1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}


def serve(
    args: argparse.Namespace,
) -> None:
    _LOGGER.info("Binding to '%s'", args.bindAddress)
    workers = []
    providers = ["CPUExecutionProvider"]
    if args.gpu:
        providers = ["CUDAExecutionProvider"] + providers
    for _ in range(args.jobs):
        worker = multiprocessing.Process(
            target=_asyncRunServer,
            args=(
                args.bindAddress,
                args.model,
                providers,
                Language.parse(args.language),
                args.vocabulary,
                args.formatter,
                args.jobs,
            ),
        )
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()


def _asyncRunServer(
    bindAddress: str,
    model: str,
    providers: str,
    language: Language,
    vocabulary: Optional[str],
    formatter: Optional[str],
    jobs: int,
) -> None:
    asyncio.run(
        _runServer(bindAddress, model, providers, language, vocabulary, formatter, jobs)
    )


async def _runServer(
    bindAddress: str,
    model: str,
    providers: str,
    language: Language,
    vocabularyPath: Optional[str],
    formatterPath: Optional[str],
    jobs: int,
) -> None:
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=jobs),
        options=(("grpc.so_reuseport", 1),),
    )
    _addRecognizerService(
        server, model, providers, language, vocabularyPath, formatterPath
    )
    _addHealthCheckService(server, jobs)
    server.add_insecure_port(bindAddress)
    _LOGGER.info(f"Server listening on {bindAddress}")
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
        type=int,
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
        default=3,
        help="Give more output. Option is additive, and can be used up to 4 times.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.set_start_method(
        "spawn"
    )  # This is important and MUST be inside the name==main block
    args = _parseArguments()
    logging.basicConfig(
        level=_LOG_LEVELS.get(args.verbose, logging.INFO),
        format="[%(asctime)s.%(msecs)03d %(levelname)s %(module)s::%(funcName)s] (PID %(process)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not Language.check(args.language):
        raise ValueError(f"Invalid language '{args.language}'")
    serve(args)
