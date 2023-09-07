import os
import grpc
import wave
import time
import toml
import jiwer
import random
import logging
import pytest
import asyncio
import unittest
import tempfile
import multiprocessing
from grpc import _channel
from concurrent import futures
from typing import Optional, Iterator

from asr4_streaming.recognizer import Server
from asr4_streaming.recognizer import RecognizerStub
from asr4_streaming.recognizer import RecognizerService
from asr4_streaming.recognizer import StreamingRecognizeRequest
from asr4_streaming.recognizer import StreamingRecognizeResponse
from asr4_streaming.recognizer import StreamingRecognitionResult
from asr4_streaming.recognizer import add_RecognizerServicer_to_server
from asr4_streaming.recognizer_v1.types import RecognitionAlternative

from google.protobuf.duration_pb2 import Duration
from grpc_health.v1.health_pb2_grpc import HealthStub
from grpc_health.v1.health_pb2 import HealthCheckRequest
from grpc_health.v1.health_pb2 import HealthCheckResponse

from tests.unit.test_event_handler import streamingRequestIterator


def initializeTomlConfig(tomlPath: str, **kwargs) -> str:
    fp = tempfile.NamedTemporaryFile(delete=False)
    tomlConfiguration = toml.load(tomlPath)
    for k, v in kwargs.items():
        tomlConfiguration["global"].setdefault(k, v)
    tomlConfiguration = toml.dumps(tomlConfiguration)
    fp.write(tomlConfiguration.encode())
    fp.seek(0)
    fp.close()
    return fp.name


@pytest.mark.usefixtures("datadir")
class RecognizerServiceTestCase(unittest.TestCase):
    _serverAddress = "localhost:8000"
    _language = "en-us"
    _kwargs = {}

    @classmethod
    def setUpClass(cls):
        multiprocessing.set_start_method("spawn", force=True)
        tomlPath = (
            os.path.dirname(os.path.realpath(__file__))
            + f"/data/asr4_streaming_config_{cls._language}.toml"
        )
        cls._event = multiprocessing.Event()
        cls._server = multiprocessing.Process(
            target=RecognizerServiceTestCase.runServer,
            args=(
                cls._event,
                cls._serverAddress,
                tomlPath,
            ),
            kwargs=cls._kwargs,
            daemon=True,
        )
        cls._server.start()

    @staticmethod
    def runServer(
        event: multiprocessing.Event, serverAddress: str, tomlPath: str, **kwargs
    ):
        asyncio.run(
            RecognizerServiceTestCase.runServerAsyncAndListenForStop(
                event, serverAddress, tomlPath, **kwargs
            )
        )

    @staticmethod
    async def runServerAsyncAndListenForStop(
        event: multiprocessing.Event, serverAddress: str, tomlPath: str, **kwargs
    ):
        server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=1))
        loop = asyncio.get_event_loop()

        serverTask = asyncio.create_task(
            RecognizerServiceTestCase.runServerAsync(
                server, serverAddress, tomlPath, **kwargs
            )
        )
        await loop.run_in_executor(None, event.wait)
        await server.stop(None)
        await serverTask

    @staticmethod
    async def runServerAsync(
        server: grpc.aio.Server, serverAddress: str, tomlPath: str, **kwargs
    ):
        tomlConfiguration = initializeTomlConfig(tomlPath, **kwargs)
        service = RecognizerService(tomlConfiguration)
        add_RecognizerServicer_to_server(service, server)
        Server._addHealthCheckService(server, jobs=1)
        server.add_insecure_port(serverAddress)
        os.unlink(tomlConfiguration)
        await server.start()
        await server.wait_for_termination()

    @classmethod
    def tearDownClass(cls):
        cls._event.set()
        cls._server.terminate()
        cls._server.join()
        cls._server.close()
        del cls._server

    def expectStatus(
        self, response: _channel._MultiThreadedRendezvous, statusCode: grpc.StatusCode
    ):
        self.assertEqual(response.code(), statusCode)

    def expectDetails(self, response: _channel._MultiThreadedRendezvous, details: str):
        self.assertEqual(response.details(), details)

    def expectValidConfidence(self, confidence: float):
        self.assertTrue(0.0 <= confidence <= 1.0)

    def expectFinal(self, response: StreamingRecognizeResponse, isFinal: bool = True):
        self.assertEqual(response.results.is_final, isFinal)

    def expectDuration(self, duration: Duration, seconds: int, nanos: int):
        self.assertEqual(duration.seconds, seconds)
        self.assertEqual(duration.nanos, nanos)

    def expectNotEmptyTranscription(self, response: StreamingRecognizeResponse):
        transcription = response.results.alternatives[0].transcript
        self.assertLess(0, len(transcription))

    def expectNumberOfWords(
        self,
        response: StreamingRecognizeResponse,
        numberOfWords: int,
        delta: float = 0.0,
    ):
        transcription = response.results.alternatives[0].transcript.split(" ")
        self.assertAlmostEqual(len(transcription), numberOfWords, delta=delta)

    def expectExactTranscription(
        self, response: StreamingRecognizeResponse, expectedResponse: str
    ):
        self.assertEquals(response.results.alternatives[0].transcript, expectedResponse)

    def expectCapitalization(self, response: StreamingRecognizeResponse):
        self.assertTrue(response.results.alternatives[0].transcript[0].isupper())

    def expectTranscriptionWER(
        self,
        response: StreamingRecognizeResponse,
        expectedResponse: str,
        expectedWER: float,
        delta: float = 0.0,
    ):
        self.assertAlmostEqual(
            jiwer.wer(response.results.alternatives[0].transcript, expectedResponse),
            expectedWER,
            delta=delta,
        )

    def expectTranscriptionCER(
        self,
        response: StreamingRecognizeResponse,
        expectedResponse: str,
        expectedCER: float,
        delta: float = 0.0,
    ):
        self.assertAlmostEqual(
            jiwer.cer(response.results.alternatives[0].transcript, expectedResponse),
            expectedCER,
            delta=delta,
        )

    def expectValidWords(self, response: StreamingRecognizeResponse):
        for word, expectedWord in zip(
            response.results.alternatives[0].words,
            response.results.alternatives[0].transcript.split(),
        ):
            self.assertEqual(word.word, expectedWord)
            self.expectValidConfidence(word.confidence)

    def expectValidWordTimestamps(
        self, response: StreamingRecognizeResponse, audioDuration: float
    ):
        for idx, word in enumerate(response.results.alternatives[0].words):
            startTime = word.start_time.ToTimedelta().total_seconds()
            endTime = word.end_time.ToTimedelta().total_seconds()
            if not idx:
                self.assertGreaterEqual(startTime, 0.0)
            else:
                self.assertGreater(startTime, 0.0)
                self.assertGreaterEqual(
                    endTime,
                    response.results.alternatives[0]
                    .words[idx - 1]
                    .start_time.ToTimedelta()
                    .total_seconds(),
                )
            self.assertGreater(endTime, startTime)
            self.assertGreaterEqual(audioDuration, endTime)

    def request(
        self,
        audioPath: str,
        language: str,
        timeout: Optional[int] = None,
        alternativeSampleRate: Optional[int] = None,
    ) -> Iterator[StreamingRecognizeResponse]:
        self._waitForServer()
        try:
            channel = grpc.insecure_channel(self._serverAddress)
            audioFile = os.path.join(self.datadir, audioPath)
            return RecognizerStub(channel).StreamingRecognize(
                RecognizerServiceTestCase.streamingRequestIteratorFromAudio(
                    audioFile, language, alternativeSampleRate
                ),
                timeout=timeout,
            )
        except Exception as e:
            self.fail(str(e))

    def mergeAllResponsesIntoOne(
        self, responseIterator: Iterator[StreamingRecognizeResponse]
    ) -> StreamingRecognizeResponse:
        responsesNum = 0
        duration, endTime = Duration(), Duration()
        mergedAlternative = RecognitionAlternative()
        for response in responseIterator:
            responsesNum += 1
            endTime = response.results.end_time
            duration = response.results.duration
            mergedAlternative.transcript += response.results.alternatives[0].transcript
            mergedAlternative.confidence += response.results.alternatives[0].confidence
            mergedAlternative.words.extend(response.results.alternatives[0].words)
        mergedAlternative.confidence /= responsesNum
        return StreamingRecognizeResponse(
            results=StreamingRecognitionResult(
                alternatives=[mergedAlternative],
                duration=duration,
                end_time=endTime,
                is_final=True,
            )
        )

    @staticmethod
    def streamingRequestIteratorFromAudio(
        audioFile: str, language: str, alternativeSampleRate: Optional[int] = None
    ) -> Iterator[StreamingRecognizeRequest]:
        audioChunks = []
        with wave.open(audioFile, "rb") as wav:
            frames = wav.getnframes()
            sampleRate = wav.getframerate()
            ms100, ms500 = 0.1 * sampleRate, 0.5 * sampleRate
            totalRead = 0
            while totalRead < frames:
                framesToRead = random.randint(ms100, ms500)
                audioChunks.append(wav.readframes(framesToRead))
                totalRead += framesToRead

            yield from streamingRequestIterator(
                language=language,
                sampleRate=alternativeSampleRate or sampleRate,
                audio=audioChunks,
            )

    def _waitForServer(self):
        response = None
        sleepPeriod = 0.5
        trials = 200
        for _ in range(trials):
            response = self.__pingHealth()
            if response.status == HealthCheckResponse.ServingStatus.SERVING:
                return
            time.sleep(sleepPeriod)
        self.reportServerNotRunning(response, trials * sleepPeriod)

    def __pingHealth(self) -> HealthCheckResponse:
        try:
            channel = grpc.insecure_channel(self._serverAddress)
            return HealthStub(channel).Check(
                HealthCheckRequest(service="asr4.recognizer.v1.Recognizer"), timeout=1
            )
        except Exception as e:
            logging.info("Server not available: " + str(e))
            return HealthCheckResponse(status=HealthCheckResponse.ServingStatus.UNKNOWN)

    @staticmethod
    def reportServerNotRunning(
        response: Optional[HealthCheckResponse], elapsedTime: float
    ):
        if response is None:
            raise (Exception(f"Server is not available after {elapsedTime} seconds"))
        else:
            raise (
                Exception(
                    "Server is not healthy: "
                    + HealthCheckResponse.ServingStatus.Name(response.status)
                )
            )

    @pytest.fixture(autouse=True)
    def datadir(self, pytestconfig):
        self.datadir = f"{pytestconfig.rootdir}/tests/functional/data"
