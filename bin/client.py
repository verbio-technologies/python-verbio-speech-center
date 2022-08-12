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

_workerChannelSingleton = None
_workerStubSingleton = None


def main() -> None:
    args = _parseArguments()
    responses = _process(args.server_address)
    _LOGGER.info(f"Returned responses: {responses}")


def _parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A Speech Recognition client.")
    parser.add_argument(
        "server_address", help="The address of the server (e.g. localhost:50051)"
    )
    return parser.parse_args()


def _process(serverAddress: str) -> List[RecognizeResponse]:
    workerPool = multiprocessing.Pool(
        processes=_PROCESS_COUNT,
        initializer=_initializeWorker,
        initargs=(serverAddress,),
    )
    responses = [workerPool.apply(_runWorkerQuery)]
    return list(map(RecognizeResponse.FromString, responses))


def _initializeWorker(serverAddress: str):
    global _workerChannelSingleton  # pylint: disable=global-statement
    global _workerStubSingleton  # pylint: disable=global-statement
    _LOGGER.info("Initializing worker process.")
    _workerChannelSingleton = grpc.insecure_channel(
        serverAddress, options=CHANNEL_OPTIONS
    )
    _workerStubSingleton = RecognizerStub(_workerChannelSingleton)
    atexit.register(_shutdownWorker)


def _shutdownWorker():
    _LOGGER.info("Shutting worker process down.")
    if _workerChannelSingleton is not None:
        _workerStubSingleton.stop()


def _runWorkerQuery() -> bytes:
    request = RecognizeRequest(
        config=RecognitionConfig(
            parameters=RecognitionParameters(language="en-US", sample_rate_hz=8000),
            resource=RecognitionResource(topic="GENERIC"),
        ),
        audio=b"",
    )
    _LOGGER.info("Running recognition.")
    return _workerStubSingleton.Recognize(request, timeout=10).SerializeToString()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d %(levelname)s %(module)s::%(funcName)s] (PID %(process)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main()
