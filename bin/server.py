import grpc
import asyncio
import logging
import multiprocessing
from concurrent import futures

from asr4.recognizer import RecognizerService
from asr4.recognizer import add_RecognizerServicer_to_server


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


def _asyncRunServer(bindAddress: str):
    asyncio.run(_runServer(bindAddress))


async def _runServer(bindAddress: str):
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=_THREAD_CONCURRENCY),
        options=(("grpc.so_reuseport", 1),),
    )
    add_RecognizerServicer_to_server(RecognizerService(), server)
    server.add_insecure_port(bindAddress)
    _LOGGER.info(f"Server listening on {bindAddress}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d %(levelname)s %(module)s::%(funcName)s] (PID %(process)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    serve()
