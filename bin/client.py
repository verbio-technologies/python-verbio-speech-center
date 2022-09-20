import sys
import grpc
import wave
import atexit
import resampy
import logging
import argparse
import multiprocessing
import numpy as np

from typing import List

from asr4.recognizer import Language
from asr4.recognizer import RecognizerStub
from asr4.recognizer import RecognizeRequest
from asr4.recognizer import RecognizeResponse
from asr4.recognizer import RecognitionConfig
from asr4.recognizer import RecognitionParameters
from asr4.recognizer import RecognitionResource


_LOGGER = logging.getLogger(__name__)
_PROCESS_COUNT = 8
_LOG_LEVELS = {1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
CHANNEL_OPTIONS = [
    ("grpc.lb_policy_name", "pick_first"),
    ("grpc.enable_retries", 0),
    ("grpc.keepalive_timeout_ms", 10000),
]

_workerChannelSingleton = None
_workerStubSingleton = None


def _process(args: argparse.Namespace) -> List[RecognizeResponse]:
    audio = _getAudio(args.audio)
    workerPool = multiprocessing.Pool(
        processes=args.jobs,
        initializer=_initializeWorker,
        initargs=(args.host,),
    )
    responses = [
        workerPool.apply(
            _runWorkerQuery,
            (
                audio,
                args.language,
            ),
        )
    ]
    return list(map(RecognizeResponse.FromString, responses))


def _getAudio(audio_file: str) -> bytes:
    with wave.open(audio_file) as f:
        n = f.getnframes()
        audio = f.readframes(n)
        sample_rate = f.getframerate()
    audio = np.frombuffer(audio, dtype=np.int16)
    y = resampy.resample(audio, sample_rate, 16000)
    audio = y.astype(audio.dtype)
    return audio.tobytes()


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


def _runWorkerQuery(audio: bytes, language: Language) -> bytes:
    request = RecognizeRequest(
        config=RecognitionConfig(
            parameters=RecognitionParameters(
                language=language.value, sample_rate_hz=16000
            ),
            resource=RecognitionResource(topic="GENERIC"),
        ),
        audio=audio,
    )
    _LOGGER.info(
        "Running recognition. "
        "If the length of the audio is one minute or more, the process may take several seconds to complete. "
    )
    try:
        return _workerStubSingleton.Recognize(
            request, metadata=(("accept-language", language.value),), timeout=60
        ).SerializeToString()
    except Exception as e:
        _LOGGER.error(f"Error in gRPC Call: {e.details()} [status={e.code()}]")
        return b""


def _parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A Speech Recognition client.")
    parser.add_argument(
        "-a",
        "--audio-path",
        required=True,
        dest="audio",
        help="Path to the audio file.",
    )
    parser.add_argument(
        "-l",
        "--language",
        dest="language",
        default=Language.EN_US.value,
        choices=[l.value for l in Language],
        help="Language of the recognizer service.",
    )
    parser.add_argument(
        "--host", default="localhost:50051", help="Hostname address of the ASR4 server."
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
    args = _parseArguments()
    logging.basicConfig(
        level=_LOG_LEVELS.get(args.verbose, logging.INFO),
        format="[%(asctime)s.%(msecs)03d %(levelname)s %(module)s::%(funcName)s] (PID %(process)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    responses = _process(args)
    _LOGGER.info(f"Returned responses: {responses}")
