import grpc
import asyncio
import logging
import multiprocessing
from concurrent import futures

from asr4.recognizer import SERVICES_NAMES
from asr4.recognizer import RecognizerService
from asr4.recognizer import add_RecognizerServicer_to_server

from grpc_health.v1 import health
from grpc_health.v1.health_pb2 import HealthCheckResponse
from grpc_health.v1.health_pb2_grpc import add_HealthServicer_to_server


_LOGGER = logging.getLogger(__name__)
_PROCESS_COUNT = multiprocessing.cpu_count()
_THREAD_CONCURRENCY = _PROCESS_COUNT


def serve(bindAddress: str = "[::]:50051") -> None:
    _LOGGER.info("Binding to '%s'", bindAddress)
    workers = []
    for _ in range(_PROCESS_COUNT):
        worker = multiprocessing.Process(target=_asyncRunServer, args=(bindAddress,))
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()


def _asyncRunServer(bindAddress: str) -> None:
    asyncio.run(_runServer(bindAddress))


async def _runServer(bindAddress: str) -> None:
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=_THREAD_CONCURRENCY),
        options=(("grpc.so_reuseport", 1),),
    )
    _addRecognizerService(server)
    _addHealthCheckService(server)
    server.add_insecure_port(bindAddress)
    _LOGGER.info(f"Server listening on {bindAddress}")
    await server.start()
    await server.wait_for_termination()


def _addRecognizerService(server: grpc.aio.Server) -> None:
    add_RecognizerServicer_to_server(RecognizerService(), server)


def _addHealthCheckService(server: grpc.aio.Server) -> None:
    healthServicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=futures.ThreadPoolExecutor(
            max_workers=_THREAD_CONCURRENCY
        ),
    )
    _markAllServicesAsHealthy(healthServicer)
    add_HealthServicer_to_server(healthServicer, server)


def _markAllServicesAsHealthy(healthServicer: health.HealthServicer) -> None:
    for service in SERVICES_NAMES + [health.SERVICE_NAME]:
        healthServicer.set(service, HealthCheckResponse.SERVING)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d %(levelname)s %(module)s::%(funcName)s] (PID %(process)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    serve()
