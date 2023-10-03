import os
import re
import sys
import grpc
import wave
import math
import pause
import argparse
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
from typing import List, Tuple, Iterator

import asyncio
from subprocess import Popen
from examples import run_evaluator
from google.protobuf.json_format import MessageToJson

from asr4_streaming.recognizer import RecognizerStub
from asr4_streaming.recognizer import RecognitionConfig
from asr4_streaming.recognizer import RecognitionResource
from asr4_streaming.recognizer import RecognitionParameters
from asr4_streaming.recognizer import StreamingRecognizeRequest
from asr4_streaming.recognizer import StreamingRecognizeResponse

from asr4_engine.data_classes import Language

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
_DEFAULT_CHUNK_SIZE = 4000


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
        self._grpcChunkSize = args.chunkSize
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
        audios = self.__getAudios()
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
            response = self.__mergeAllStreamResponsesIntoOne(responses)
            self._listStreamingRecognizeResponses.append(response)
            self._trnHypothesis.append(self.__getTrnHypothesis(response, audioPath))
        logger.debug(f'[-] TRN Hypothesis: "{self._trnHypothesis}')

    def __getAudios(self) -> List[str]:
        audios = self.__getAudiosListFromGUI(self._gui) if self._gui else [self._audio]
        logger.debug(f"- Read {len(audios)} files from GUI.")
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
                request,
                metadata=(
                    ("accept-language", language.value),
                    ("user-id", "testUser"),
                    ("request-id", "testRequest"),
                ),
                timeout=20 * audioDuration,
            )
        except grpc.RpcError as e:
            logger.error(f"Error in gRPC Call: {e.details()} [status={e.code()}]")
        except Exception as e:
            logger.error(f"Error in gRPC Call: {e}")

    async def _readResponseFromStream(self) -> List[StreamingRecognizeResponse]:
        response = []
        try:
            async for chunk in self.grpcResponseStream:
                logger.debug(
                    f" - Got stream response {len(response)} > {chunk.results.alternatives[0].transcript}"
                )
                response.append(chunk)
        except grpc.RpcError as e:
            logger.error(f"Error in gRPC Call: {e.details()} [status={e.code()}]")
        except Exception as e:
            logger.error(f"Error in gRPC Call: {e}")
        return response

    def __createStreamingRequests(
        self,
        audio: bytes,
        sampleRateHz: int,
        sampleWidth: int,
        language: Language,
        useFormat: bool,
        batchMode: bool,
    ) -> Iterator[StreamingRecognizeRequest]:
        chunkSize = self.__setChunkSize(batchMode)
        chunkDuration = chunkSize / (sampleWidth * sampleRateHz)
        yield StreamingRecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(
                    language=language.value,
                    sample_rate_hz=sampleRateHz,
                    enable_formatting=useFormat,
                ),
                resource=RecognitionResource(topic="GENERIC"),
            )
        )
        yield from self.__yieldAudioSegmentsInStream(audio, chunkSize, chunkDuration)

    def __yieldAudioSegmentsInStream(
        self,
        audio: bytes,
        chunkSize: int,
        chunkDuration: float,
    ) -> Iterator[StreamingRecognizeRequest]:
        messages = self.__getAudioStreamingRequests(audio, chunkSize)
        messageNum = math.ceil(len(audio) / chunkSize) if chunkSize else 1
        for n, message in enumerate(messages, start=1):
            getUpTime = datetime.now() + timedelta(seconds=chunkDuration)
            logger.debug(f"Sending stream message {n} of {messageNum}")
            yield message
            pause.until(getUpTime)

    def __getAudioStreamingRequests(
        self, audio: bytes, chunkSize: int
    ) -> Iterator[StreamingRecognizeRequest]:
        if not audio:
            logger.error(f"Empty audio content: {audio}")
        else:
            chunkSize = chunkSize or len(audio)
            for i in range(0, len(audio), chunkSize):
                yield StreamingRecognizeRequest(audio=audio[i : i + chunkSize])

    def __setChunkSize(self, batchMode: bool) -> int:
        if batchMode:
            logger.info(
                "Audio chunk size for gRPC channel set to 0. Uploading all the audio at once"
            )
            return 0
        else:
            return self._grpcChunkSize

    def __calculateTotalDuration(
        self, audio: bytes, sampleRateHz: int, sampleWidth: int
    ) -> int:
        audioDuration = len(audio) / (sampleWidth * sampleRateHz)
        return audioDuration

    def __mergeAllStreamResponsesIntoOne(
        self, responses: List[StreamingRecognizeResponse]
    ) -> StreamingRecognizeResponse:
        if responses:
            logger.debug(f"Joining {len(responses)} partial responses")
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
        return StreamingRecognizeResponse()

    def __getTrnHypothesis(
        self, response: StreamingRecognizeRequest, audioPath: str
    ) -> str:
        filename = re.sub(r"(.*)\.wav$", r"\1", audioPath)
        if len(response.results.alternatives) > 0:
            return f"{response.results.alternatives[0].transcript.strip()} ({filename})"
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
        help="Size (in bytes) of the data chunks send in gRPC communication.",
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


def repr(responses: List[StreamingRecognizeRequest]) -> List[str]:
    return [
        f'<StreamingRecognizeRequest first alternative: "{r.results.alternatives[0].transcript}">'
        for r in responses
        if len(r.results.alternatives) > 0
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
