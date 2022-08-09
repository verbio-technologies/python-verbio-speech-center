import sys
import grpc
import atexit
import logging
import argparse
import multiprocessing

from typing import List

from asr4.recognizer import RecognizerStub
from asr4.recognizer import RecognizeRequest
from asr4.recognizer import RecognizeResponse
from asr4.recognizer import RecognitionConfig
from asr4.recognizer import RecognitionParameters
from asr4.recognizer import RecognitionResource


_LOGGER = logging.getLogger(__name__)
_PROCESS_COUNT = 8
CHANNEL_OPTIONS = [
    ("grpc.lb_policy_name", "pick_first"),
    ("grpc.enable_retries", 0),
    ("grpc.keepalive_timeout_ms", 10000),
]

_worker_channel_singleton = None
_worker_stub_singleton = None


def main() -> None:
    args = _parse_arguments()
    responses = _process(args.server_address)
    _LOGGER.info(f"Returned responses: {responses}")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A Speech Recognition client.")
    parser.add_argument(
        "server_address", help="The address of the server (e.g. localhost:50051)"
    )
    return parser.parse_args()


def _process(server_address: str) -> List[RecognizeResponse]:
    worker_pool = multiprocessing.Pool(
        processes=_PROCESS_COUNT,
        initializer=_initialize_worker,
        initargs=(server_address,),
    )
    responses = [worker_pool.apply(_run_worker_query)]
    return list(map(RecognizeResponse.FromString, responses))


def _initialize_worker(server_address: str):
    global _worker_channel_singleton  # pylint: disable=global-statement
    global _worker_stub_singleton  # pylint: disable=global-statement
    _LOGGER.info("Initializing worker process.")
    _worker_channel_singleton = grpc.insecure_channel(
        server_address, options=CHANNEL_OPTIONS
    )
    _worker_stub_singleton = RecognizerStub(_worker_channel_singleton)
    atexit.register(_shutdown_worker)


def _shutdown_worker():
    _LOGGER.info("Shutting worker process down.")
    if _worker_channel_singleton is not None:
        _worker_channel_singleton.stop()


def _run_worker_query() -> bytes:
    request = RecognizeRequest(
        config=RecognitionConfig(
            parameters=RecognitionParameters(language="en-US", sample_rate_hz=8000),
            resource=RecognitionResource(topic="GENERIC"),
        ),
        audio=b"",
    )
    _LOGGER.info("Running recognition.")
    return _worker_stub_singleton.Recognize(request, timeout=10).SerializeToString()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d %(levelname)s %(module)s::%(funcName)s] (PID %(process)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
