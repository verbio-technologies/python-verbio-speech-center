import argparse
import asyncio
import atexit
import grpc
import math
import numpy as np
import os
import pause
import re
import sys
import wave
from loguru import logger
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Iterator

import multiprocessing
from subprocess import Popen
from examples import run_evaluator
from google.protobuf.json_format import MessageToJson

from asr4_streaming.recognizer import RecognizerStub
from asr4_streaming.recognizer import StreamingRecognizeRequest
from asr4_streaming.recognizer import StreamingRecognizeResponse
from asr4_streaming.recognizer import StreamingRecognitionResult
from asr4_streaming.recognizer import RecognitionConfig
from asr4_streaming.recognizer import RecognitionParameters
from asr4_streaming.recognizer import RecognitionResource

from asr4.engines.wav2vec.v1.engine_types import Language

_PROCESS_COUNT = 8

_LOG_LEVELS = {
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "TRACE",
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
_DEFAULT_CHUNK_SIZE = 20_000


class StreamingClient:
    def __init__(self, args: argparse.Namespace):
        self._gui = args.gui
        self._jobs = args.jobs
        self._host = args.host
        self._batch = args.batch
        self._audio = args.audio
        self._format = args.format
        self._language = args.language
        self._trnHypothesis: List[str] = []
        self._listStreamingRecognizeResponses: List[StreamingRecognizeResponse] = []
        self.grpcResponseStream = None

    async def process(self) -> Tuple[List[str], List[StreamingRecognizeResponse]]:
        await self.__inferenceProcess()
        logger.debug(
            f"[+] Generating Responses from {len(self._listStreamingRecognizeResponses)} candidates"
        )
        return (
            self._trnHypothesis,
            self._listStreamingRecognizeResponses,
        )

    async def __inferenceProcess(self):
        responses = []
        audios = self.__getAudios()
        logger.debug("- Read %d files from GUI." % len(audios))
        self.__initializeWorker(self._host)
        for n, audioPath in enumerate(audios):
            audio, sampleRateHz, sampleWidth = self.__getAudio(audioPath)
            _, responses = await asyncio.gather(
                self._runWorkerQuery(
                    audio,
                    sampleRateHz,
                    sampleWidth,
                    Language.parse(self._language),
                    self._format,
                    f"{n}/{len(audios)}",
                    self._batch,
                ),
                self._readResponseFromStream(),
            )
            response = self.mergeStreamingResults(responses)
            self._listStreamingRecognizeResponses.append(response)
            self._trnHypothesis.append(self.__getTrnHypothesis(response, audioPath))
            responses.append(response)
        logger.debug(f'[-] TRN Hypothesis: "{self._trnHypothesis}')
        return responses

    def mergeStreamingResults(
        self, chunks: List[StreamingRecognizeResponse]
    ) -> StreamingRecognizeResponse:
        logger.debug(f"Joining {len(chunks)} partial responses")
        merged = chunks.pop(0).results
        for chunk in chunks:
            merged.alternatives[0].transcript += (
                " " + chunk.results.alternatives[0].transcript
            )
            for word in chunk.results.alternatives[0].words:
                merged.alternatives[0].words.append(word)
            merged.end_time.CopyFrom(chunk.results.end_time)
            merged.duration.CopyFrom(chunk.results.end_time)
        return merged

    def __getAudios(self):
        audios = []
        if self._gui:
            audios = self.__getAudiosListFromGUI(self._gui)
        else:
            audios.append(self._audio)
        return audios

    def __getAudiosListFromGUI(self, gui: str) -> List[str]:
        return [audio for audio in open(gui, "r").read().split("\n") if audio != ""]

    def __getAudio(self, audioFile: str) -> Tuple[bytes, int, int]:
        with wave.open(audioFile) as f:
            n = f.getnframes()
            audio = f.readframes(n)
            sampleRateHz = f.getframerate()
            sampleWidth = f.getsampwidth()
        self.__checkSampleValues(audioFile, sampleWidth)
        audio = np.frombuffer(audio, dtype=np.int16)
        return (audio.tobytes(), sampleRateHz, sampleWidth)

    def __checkSampleValues(self, fileName: str, sampleWidth: int):
        if sampleWidth != 2:
            raise Exception(
                f"Error, audio file {fileName} should have 2-byte samples instead of {sampleWidth}-byte samples."
            )

    def __initializeWorker(self, serverAddress: str):
        global _workerChannelSingleton  # pylint: disable=global-statement
        global _workerStubSingleton  # pylint: disable=global-statement
        logger.info("Initializing worker process.")
        _workerChannelSingleton = grpc.aio.insecure_channel(
            serverAddress, options=CHANNEL_OPTIONS
        )
        _workerStubSingleton = RecognizerStub(_workerChannelSingleton)
        atexit.register(self.__shutdownWorker)

    def __shutdownWorker(self):
        logger.info("Shutting worker process down.")

    def __createStreamingRequests(
        self,
        audio: bytes,
        sampleRateHz: int,
        sampleWidth: int,
        language: Language,
        useFormat: bool,
        batchMode: bool,
    ) -> List[StreamingRecognizeRequest]:
        request = [
            StreamingRecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language=language.value,
                        sample_rate_hz=sampleRateHz,
                        enable_formatting=useFormat,
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                )
            )
        ]
        chunkSize = self.__setChunkSize(batchMode)
        chunkDuration = chunkSize / (sampleWidth * sampleRateHz)
        yield from self._yieldAudioSegmentsInStream(
            request, audio, chunkSize, chunkDuration
        )

    def _yieldAudioSegmentsInStream(self,
                                    request: List[StreamingRecognizeRequest],
                                    audio: bytes,
                                    chunkSize: int,
                                    chunkDuration: float ):
        messages = self._addAudioSegmentsToStreamingRequest(request, audio, chunkSize)
        for n, message in enumerate(messages):
            getUpTime = datetime.now() + timedelta(seconds=chunkDuration)
            logger.debug(f"Sending stream message {n} of {len(messages)-1}")
            yield message
            pause.until(getUpTime)

    def _addAudioSegmentsToStreamingRequest(self, request, audio, chunkSize):
        for chunk in self.__chunk_audio(audio=audio, chunkSize=chunkSize):
            if chunk != []:
                request.append(StreamingRecognizeRequest(audio=chunk))
        return request

    def __chunk_audio(self, audio: bytes, chunkSize: int):
        if not audio:
            logger.error(f"Empty audio content: {audio}")
            yield audio
        else:
            if chunkSize == 0:
                logger.info(
                    "Audio chunk size for gRPC channel set to 0. Uploading all the audio at once"
                )
                yield audio
            else:
                for i in range(0, len(audio), chunkSize):
                    yield audio[i : i + chunkSize]

    async def _runWorkerQuery(
        self,
        audio: bytes,
        sampleRateHz: int,
        sampleWidth: int,
        language: Language,
        useFormat: bool,
        queryID: int,
        batchMode: bool,
    ):
        audioDuration = self.__calculateTotalDuration(audio, sampleRateHz, sampleWidth)
        request = self.__createStreamingRequests(
            audio, sampleRateHz, sampleWidth, language, useFormat, batchMode
        )
        logger.info(
            f"Running recognition {queryID}. May take several seconds for audios longer that one minute."
        )
        try:
            self.grpcResponseStream = _workerStubSingleton.StreamingRecognize(
                iter(request),
                metadata=(("accept-language", language.value),),
                timeout=20 * audioDuration,
            )
        except grpc.RpcError as e:
            logger.error(f"Error in gRPC Call: {e.details()} [status={e.code()}]")
        except Exception as e:
            logger.error(f"Error in gRPC Call: {e}")

    async def _readResponseFromStream(self) -> List[StreamingRecognizeResponse]:
        response = []
        n = 0
        async for chunk in self.grpcResponseStream:
            logger.debug(
                f" - Got stream response {n} > {chunk.results.alternatives[0].transcript}"
            )
            response.append(chunk)
            n += 1
        return response

    def __setChunkSize(self, batchMode: bool) -> int:
        if batchMode:
            logger.info(
                "Audio chunk size for gRPC channel set to 0. Uploading all the audio at once"
            )
            return 0
        else:
            return _DEFAULT_CHUNK_SIZE

    def __calculateTotalDuration(
        self, audio: bytes, sampleRateHz: int, sampleWidth: int
    ) -> int:
        audioDuration = len(audio) / (sampleWidth * sampleRateHz)
        return audioDuration

    def __mergeAllStreamResponsesIntoOne(
        self, responses: List[StreamingRecognizeResponse]
    ) -> Optional[StreamingRecognizeResponse]:
        if responses:
            for response in responses[1:]:
                responses[0].results.alternatives[0].transcript += (
                    " " + response.results.alternatives[0].transcript
                )
                responses[0].results.alternatives[0].words.extend(
                    response.results.alternatives[0].words
                )
            responses[0].results.duration.CopyFrom(responses[-1].results.end_time)
            responses[0].results.end_time.CopyFrom(responses[-1].results.end_time)
            return responses[0]
        return None

    def __getTrnHypothesis(
        self, response: StreamingRecognitionResult, audioPath: str
    ) -> str:
        filename = re.sub(r"(.*)\.wav$", r"\1", audioPath)
        if len(response.alternatives) > 0:
            return f"{response.alternatives[0].transcript.strip()} ({filename})"
        else:
            return f" ({filename})"

def getMetrics(args: argparse.Namespace, trnHypothesis: List[str]) -> Popen:
    logger.trace("Running evaluation.")
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    popenArgs = [
        "python3",
        (run_evaluator.__file__),
        "--hypothesis",
        generateTrnFile(args, trnHypothesis, "trnHypothesis.trn"),
        "--reference",
        generateTrnReferencesFile(args),
        "--output",
        args.output,
        "--language",
        args.language,
        "--encoding",
        _ENCODING,
        "--test_id",
        "test_" + args.language,
    ]
    Popen(
        popenArgs,
        stdout=sys.stdout,
        stderr=sys.stderr,
        universal_newlines=True,
    )
    logger.info("You can find the files of results in path: " + args.output)


def _generateTrnHypothesisFile(
    self, args: argparse.Namespace, trnHypothesis: List[str]
) -> str:
    trnHypothesisFile = os.path.join(args.output, "trnHypothesis.trn")
    with open(trnHypothesisFile, "w") as h:
        h.write("\n".join(trnHypothesis))
    return trnHypothesisFile

def generateTrnFile(
    args: argparse.Namespace, trnHypothesis: List[str], filename: str
) -> str:
    trnFile = os.path.join(args.output, filename)
    with open(trnFile, "w") as h:
        h.write("\n".join(trnHypothesis))
    return trnFile


def generateTrnReferencesFile(args: argparse.Namespace) -> str:
    trnReferences = (
        getTrnReferences([l.rstrip("\n") for l in open(args.gui)])
        if args.gui
        else getTrnReferences([args.audio])
    )
    return generateTrnFile(args, trnReferences, "trnReferences.trn")

def getTrnReferences(references: List[str]) -> List[str]:
    trnReferences = []
    for line in references:
        referenceFile = re.sub(r"(.*)\.wav$", r"\1.txt", line)
        trnReferences.append(
            open(referenceFile, "r").read().replace("\n", " ")
            + " ("
            + referenceFile.replace(".txt", "")
            + ")"
        )
    trnReferences.append("")
    return trnReferences

            

def parseArguments() -> argparse.Namespace:
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
        "--format",
        dest="format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically improve the format of the recognized text.",
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
        "--json",
        action="store_true",
        default=False,
        help="Print gRPC response messages in json format.",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        action="store_true",
        help="Calculate metrics using the audio transcription references.",
    )
    parser.add_argument(
        "-b",
        "--batch",
        action="store_true",
        help="Sent all audio at once instead of streaming.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        choices=list(_LOG_LEVELS),
        default=os.environ.get("LOG_LEVEL", _LOG_LEVEL),
        help="Log levels. Options: CRITICAL, ERROR, WARNING, INFO and DEBUG. By default reads env variable LOG_LEVEL.",
    )
    parser.add_argument(
        "-c",
        "--chuk-size",
        type=int,
        dest="chunkSize",
        default=_DEFAULT_CHUNK_SIZE,
        help="Size (in bytes) of the ",
    )
    return parser.parse_args()


def configureLogger(logLevel: str) -> None:
    logLevel = validateLogLevel(logLevel)
    logger.remove()
    logger.add(
        sys.stdout,
        level=logLevel,
        format="[{time:YYYY-MM-DDTHH:mm:ss.SSS}Z <level>{level}</level> <magenta>{module}</magenta>::<magenta>{function}</magenta>] "
        "<level>{message}</level>",
        enqueue=True,
    )

def validateLogLevel(logLevel: str) -> str:
    if logLevel not in _LOG_LEVELS:
        offender = logLevel
        logLevel = _LOG_LEVEL
        logger.warning(
            f"Level [{offender}] is not valid log level. Will use {logLevel} instead."
        )
    return logLevel

def repr(responses: List[StreamingRecognitionResult]) -> List[str]:
    return [
        f'<StreamingRecognitionResult first alternative: "{r.alternatives[0].transcript}">'
        for r in responses
        if len(r.alternatives) > 0
    ]


if __name__ == "__main__":
    args = parseArguments()
    configureLogger(args.verbose)
    if not Language.check(args.language):
        raise ValueError(f"Invalid language '{args.language}'")
    trnHypothesis, responses = asyncio.run(StreamingClient(args).process())
    logger.debug(f"Returned responses: {repr(responses)}")

    if args.metrics:
        getMetrics(args, trnHypothesis)

    if args.json:
        print("> Messages:")
        for r in responses:
            j = MessageToJson(r)
            print(j)
        print("< messages finished")
