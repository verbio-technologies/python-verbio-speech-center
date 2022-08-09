import grpc
import socket
import asyncio
import logging
import contextlib
import multiprocessing
from concurrent import futures

from asr4.recognizer import RecognizerServiceAsync
from asr4.recognizer import add_RecognizerServicer_to_server


_LOGGER = logging.getLogger(__name__)
_PROCESS_COUNT = multiprocessing.cpu_count()
_THREAD_CONCURRENCY = _PROCESS_COUNT


def serve() -> None:
    with _reserve_port() as port:
        bind_address = "localhost:{}".format(port)
        _LOGGER.info("Binding to '%s'", bind_address)
        workers = []
        for _ in range(_PROCESS_COUNT):
            worker = multiprocessing.Process(
                target=_async_run_server, args=(bind_address,)
            )
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()


@contextlib.contextmanager
def _reserve_port():
    """Find and reserve a port for all subprocesses to use."""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError("Failed to set SO_REUSEPORT.")
    sock.bind(("", 0))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def _async_run_server(bind_address: str):
    asyncio.run(_run_server(bind_address))


async def _run_server(bind_address: str):
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=_THREAD_CONCURRENCY),
        options=(("grpc.so_reuseport", 1),),
    )
    add_RecognizerServicer_to_server(RecognizerServiceAsync(), server)
    server.add_insecure_port(bind_address)
    _LOGGER.info(f"Server listening on {bind_address}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d %(levelname)s %(module)s::%(funcName)s] (PID %(process)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    serve()
