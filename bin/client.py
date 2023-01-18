import grpc
import wave
import atexit
import logging
import argparse
import multiprocessing
import numpy as np
import re
import os
import sys
import time

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

_LOG_LEVELS = {
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "TRACE": logging.DEBUG,
}
_LOG_LEVEL = "ERROR"
CHANNEL_OPTIONS = [
    ("grpc.lb_policy_name", "pick_first"),
    ("grpc.enable_retries", 0),
    ("grpc.keepalive_timeout_ms", 10000),
]

_workerChannelSingleton = None
_workerStubSingleton = None

_ENCODING = "utf-8"


def _repr(responses: List[RecognizeRequest]) -> List[str]:
    return [
        f'<RecognizeRequest first alternative: "{r.alternatives[0].transcript}">'
        for r in responses
        if len(r.alternatives) > 0
    ]


def _process(args: argparse.Namespace) -> List[RecognizeResponse]:
    responses, trnHypothesis = _inferenceProcess(args)
    trnReferences = []
    if args.metrics:
        if args.gui:
            trnReferences = _getTrnReferences(args.gui)
        else:
            referenceFile = re.sub(r"(.*)\.wav$", r"\1.txt", args.audio)
            trnReferences.append(
                open(referenceFile, "r").read().replace("\n", " ")
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

    _LOGGER.debug("[+] Generating Responses from %d candidates" % len(responses))
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
        referenceFile = re.sub(r"(.*)\.wav$", r"\1.txt", line)
        if line != "":
            try:
                reference = open(referenceFile, "r").read().replace("\n", " ")
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
    _LOGGER.debug("- Read %d files from GUI." % len(audios))

    workerPool = multiprocessing.Pool(
        processes=args.jobs,
        initializer=_initializeWorker,
        initargs=(args.host,),
    )

    for n, audio_path in enumerate(audios):
        audio, sample_rate_hz = _getAudio(audio_path)
        response = workerPool.apply(
            _runWorkerQuery,
            (
                audio,
                sample_rate_hz,
                Language.parse(args.language),
                n,
            ),
        )
        responses.append(response)
        trnHypothesis.append(_getTrnHypothesis(response, audio_path))
    trnHypothesis.append("")
    _LOGGER.debug("[-] TRN Hypothesis: {trnHypothesis}")

    return responses, trnHypothesis


def _getTrnHypothesis(response: bytes, audio_path: str) -> str:
    filename = re.sub(r"(.*)\.wav$", r"\1", audio_path)
    recognizeResponse = RecognizeResponse.FromString(response)
    if len(recognizeResponse.alternatives) > 0:
        return f"{recognizeResponse.alternatives[0].transcript} ({filename})"
    else:
        return f" ({filename})"


def _getAudiosList(gui_file: str) -> List[str]:
    return [audio for audio in open(args.gui, "r").read().split("\n") if audio != ""]


def _getAudio(audio_file: str) -> bytes:
    with wave.open(audio_file) as f:
        n = f.getnframes()
        audio = f.readframes(n)
        sample_rate_hz = f.getframerate()
    audio = np.frombuffer(audio, dtype=np.int16)
    return audio.tobytes(), sample_rate_hz


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


def _runWorkerQuery(audio: bytes, sample_rate_hz: int, language: Language, queryID: int) -> bytes:
    request = RecognizeRequest(
        config=RecognitionConfig(
            parameters=RecognitionParameters(
                language=language.value, sample_rate_hz=sample_rate_hz
            ),
            resource=RecognitionResource(topic="GENERIC"),
        ),
        audio=audio,
    )
    _LOGGER.info(
        "Running recognition {queryID}. "
        "The process may take several seconds to complete for audios longer that one minute. "
    )
    try:
        return _workerStubSingleton.Recognize(
            request, metadata=(("accept-language", language.value),), timeout=900
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
        choices=list(_LOG_LEVELS.keys()),
        default=os.environ.get("LOG_LEVEL", _LOG_LEVEL),
        help="Log levels. Options: CRITICAL, ERROR, WARNING, INFO and DEBUG. By default reads env variable LOG_LEVEL.",
    )
    return parser.parse_args()


def validateLogLevel(args):
    if args.verbose not in _LOG_LEVELS:
        offender = args.verbose
        args.verbose = _LOG_LEVEL
        _LOGGER.warning(
            "Level [%s] is not valid log level. Will use %s instead."
            % (offender, args.verbose)
        )


if __name__ == "__main__":
    args = _parseArguments()
    if not (args.audio or args.gui):
        raise ValueError(f"Audio path (-a) or audios gui file (-g) is required")
    validateLogLevel(args)
    logging.Formatter.converter = time.gmtime
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
