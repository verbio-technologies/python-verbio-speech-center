import sys
import grpc
import wave
import atexit
import resampy
import logging
import argparse
import multiprocessing
import numpy as np
from datetime import datetime
import re
import os
import sys

from subprocess import Popen, PIPE
from examples import run_evaluator

from typing import List

from asr4.types.language import Language
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

_ENCODING = "utf-8"


def _repr(responses: List[RecognizeRequest]) -> List[str]:
    return [f'<RecognizeRequest text: "{r.text}">' for r in responses]


def _process(args: argparse.Namespace) -> List[RecognizeResponse]:
    responses, trnHypothesis = _inferenceProcess(args)
    trnReferences = []
    if args.metrics:
        if args.gui:
            trnReferences = _getTrnReferences(args.gui)
        else:
            referenceFile = args.audio.replace(".wav", ".txt")
            trnReferences.append(
                open(referenceFile, "r").read()
                + " ("
                + referenceFile.replace(".txt", "")
                + ")"
            )
            trnReferences.append("")
        _getMetrics(
            trnHypothesis,
            trnReferences,
            args.output,
            "test_" + args.language,
            _ENCODING,
            args.language,
        )

    return list(map(RecognizeResponse.FromString, responses))


def _getMetrics(
    trnHypothesis: List[str],
    trnReferences: List[str],
    outputDir: str,
    id: str,
    encoding: str,
    language: str,
) -> Popen:

    _LOGGER.info("Running evaluation.")

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    trnHypothesisFile = os.path.join(outputDir, "trnHypothesis.trn")
    trnReferencesFile = os.path.join(outputDir, "trnReferences.trn")
    with open(trnHypothesisFile, "w") as h:
        h.write("\n".join(trnHypothesis))
    with open(trnReferencesFile, "w") as r:
        r.write("\n".join(trnReferences))

    Popen(
        [
            "python3",
            (run_evaluator.__file__),
            "--hypothesis",
            trnHypothesisFile,
            "--reference",
            trnReferencesFile,
            "--output",
            outputDir,
            "--language",
            language,
            "--encoding",
            encoding,
            "--test_id",
            id,
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        universal_newlines=True,
    )

    _LOGGER.info("You can find the files of results in path: " + outputDir)


def _getTrnReferences(gui: str) -> List[str]:
    trn = []
    for line in open(args.gui).read().split("\n"):
        referenceFile = line.replace(".wav", ".txt")
        if line != "":
            try:
                reference = open(referenceFile, "r").read()
            except:
                raise FileNotFoundError(f"Reference file not found.")
            trn.append(reference + " (" + referenceFile.replace(".txt", "") + ")")
            trn.append("")
    return trn


def _inferenceProcess(args: argparse.Namespace) -> List[RecognizeResponse]:
    audios = []
    responses = []
    trnHypothesis = []

    if args.gui:
        audios = _getAudiosList(args.gui)
    else:
        audios.append(args.audio)

    workerPool = multiprocessing.Pool(
        processes=args.jobs,
        initializer=_initializeWorker,
        initargs=(args.host,),
    )

    for audio_path in audios:
        audio = _getAudio(audio_path)
        response = workerPool.apply(
            _runWorkerQuery,
            (
                audio,
                Language.parse(args.language),
            ),
        )
        responses.append(response)
        trnHypothesis.append(_getTrnHypothesis(response, audio_path))
    trnHypothesis.append("")

    return responses, trnHypothesis


def _getTrnHypothesis(response: RecognizeResponse, audio_path: str) -> str:
    filename = audio_path.replace(".wav", "")
    return f"{RecognizeResponse.FromString(response).text} ({filename})"


def _getAudiosList(gui_file: str) -> List[str]:
    return [audio for audio in open(args.gui, "r").read().split("\n") if audio != ""]


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
            request, metadata=(("accept-language", language.value),), timeout=100
        ).SerializeToString()
    except Exception as e:
        _LOGGER.error(f"Error in gRPC Call: {e.details()} [status={e.code()}]")
        return b""


def _parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A Speech Recognition client.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        dest="output",
        help="Output path for the results.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-a",
        "--audio-path",
        dest="audio",
        help="Path to the audio file.",
    )
    group.add_argument(
        "-g",
        "--gui-path",
        dest="gui",
        help="Path to the gui file with audio paths.",
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
        "-m",
        "--metrics",
        action="store_true",
        help="Calculate metrics using the audio transcription references.",
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
    if not (args.audio or args.gui):
        raise ValueError(f"Audio path (-a) or audios gui file (-g) is required")
    logging.basicConfig(
        level=_LOG_LEVELS.get(args.verbose, logging.INFO),
        format="[%(asctime)s.%(msecs)03d %(levelname)s %(module)s::%(funcName)s] (PID %(process)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not Language.check(args.language):
        raise ValueError(f"Invalid language '{args.language}'")
    responses = _process(args)
    _LOGGER.info(f"Returned responses: {_repr(responses)}")
