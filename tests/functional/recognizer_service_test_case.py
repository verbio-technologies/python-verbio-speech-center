import os
import grpc
import wave
import time
import toml
import jiwer
import pause
import random
import logging
import pytest
import asyncio
import unittest
import tempfile
import multiprocessing
from grpc import _channel
from concurrent import futures
from datetime import datetime, timedelta
from typing import Optional, Iterator, Tuple, List, AsyncIterator

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
from tests.unit.test_event_handler import asyncStreamingRequestIterator


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
class RecognizerServiceTestCase(unittest.IsolatedAsyncioTestCase):
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
        cls._audioChunksIndex = 0.0
        cls._audioChunksLatency = []

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
        del cls._audioChunksIndex
        del cls._audioChunksLatency

    def expectStatus(
        self, response: _channel._MultiThreadedRendezvous, statusCode: grpc.StatusCode
    ):
        self.assertEqual(response.code(), statusCode)

    async def expectAsyncStatus(
        self, response: _channel._MultiThreadedRendezvous, statusCode: grpc.StatusCode
    ):
        self.assertEqual(await response.code(), statusCode)

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

    def expectLatency(
        self,
        response: StreamingRecognizeResponse,
        responseTime: float,
        expectedLatency: float,
    ):
        responseLatency = 0.0
        for word in response.results.alternatives[0].words:
            foundAudioChunk = False
            endTime = word.end_time.ToTimedelta().total_seconds()
            for chunk in self._audioChunksLatency:
                if endTime >= chunk["start"] and endTime <= chunk["end"]:
                    responseLatency += responseTime - chunk["epoch"]
                    foundAudioChunk = True
                    break
            if not foundAudioChunk:
                self.fail(
                    f"Could not relate a word with any audio chunk. [word={word}] [audioChunks={self._audioChunksLatency}]"
                )
        if responseLatency:
            responseLatency /= len(response.results.alternatives[0].words)
        self.assertLessEqual(responseLatency, expectedLatency)

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
                metadata=(
                    ("user-id", "testUser"),
                    ("request-id", "testRequest"),
                ),
                timeout=timeout,
            )
        except Exception as e:
            self.fail(str(e))

    async def requestAsync(
        self,
        audioPath: str,
        language: str,
    ) -> AsyncIterator[StreamingRecognizeResponse]:
        self._waitForServer()
        try:
            channel = grpc.aio.insecure_channel(self._serverAddress)
            audioFile = os.path.join(self.datadir, audioPath)
            responseIterator = RecognizerStub(channel).StreamingRecognize(
                self.__streamingAsyncRequestIteratorFromAudio(audioFile, language),
                metadata=(
                    ("user-id", "testUser"),
                    ("request-id", "testRequest"),
                ),
            )
            async for response in responseIterator:
                yield response
            await self.expectAsyncStatus(responseIterator, grpc.StatusCode.OK)
        except Exception as e:
            self.fail(str(e))

    @staticmethod
    def streamingRequestIteratorFromAudio(
        audioFile: str, language: str, alternativeSampleRate: Optional[int] = None
    ) -> Iterator[StreamingRecognizeRequest]:
        audioChunks, sampleRate = RecognizerServiceTestCase.requestsFromAudio(audioFile)
        yield from streamingRequestIterator(
            language=language,
            sampleRate=alternativeSampleRate or sampleRate,
            audio=audioChunks,
            enable_formatting=True,
        )

    async def __streamingAsyncRequestIteratorFromAudio(
        self,
        audioFile: str,
        language: str,
    ) -> AsyncIterator[StreamingRecognizeRequest]:
        audioChunks, sampleRate = RecognizerServiceTestCase.requestsFromAudio(audioFile)
        async for request in asyncStreamingRequestIterator(
            language=language,
            sampleRate=sampleRate,
            audio=audioChunks,
        ):
            if request.HasField("audio"):
                chunkDuration = len(request.audio) / 2 / sampleRate
                getUpTime = datetime.now() + timedelta(seconds=chunkDuration)
                self._audioChunksLatency.append(
                    {
                        "start": self._audioChunksIndex,
                        "end": self._audioChunksIndex + chunkDuration,
                        "epoch": time.time(),
                    }
                )
                self._audioChunksIndex += chunkDuration
            yield request
            if request.HasField("audio"):
                pause.until(getUpTime)

    @staticmethod
    def requestsFromAudio(audioFile: str) -> Tuple[List[bytes], int]:
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
            return (audioChunks, sampleRate)

    @staticmethod
    def mergeAllResponsesIntoOne(
        responseIterator: Iterator[StreamingRecognizeResponse],
    ) -> StreamingRecognizeResponse:
        responsesNum = 0
        duration, endTime = Duration(), Duration()
        mergedAlternative = RecognitionAlternative()
        for response in responseIterator:
            responsesNum += 1
            endTime = response.results.end_time
            duration = response.results.duration
            transcript = response.results.alternatives[0].transcript
            transcript = transcript if responsesNum == 1 else " " + transcript
            mergedAlternative.transcript += transcript
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
    def mergeResponsesIntoOne(
        mergedResponse: Optional[StreamingRecognizeResponse],
        response: StreamingRecognizeResponse,
    ) -> StreamingRecognizeResponse:
        if not mergedResponse:
            return response
        else:
            mergedResponse.results.end_time.CopyFrom(response.results.end_time)
            mergedResponse.results.duration.CopyFrom(response.results.end_time)
            mergedResponse.results.alternatives[0].transcript += (
                " " + response.results.alternatives[0].transcript
            )
            mergedResponse.results.alternatives[0].words.extend(
                response.results.alternatives[0].words
            )
            return mergedResponse

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
