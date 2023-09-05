import os
import grpc
import math
import wave
import pytest
import logging
from unittest.mock import patch
import asyncio
import unittest
import multiprocessing
from concurrent import futures

from asr4_streaming.recognizer import RecognizerStub
from asr4_streaming.recognizer import RecognizerService
from asr4_streaming.recognizer import StreamingRecognizeRequest
from asr4_streaming.recognizer import RecognitionConfig
from asr4_streaming.recognizer import RecognitionParameters
from asr4_streaming.recognizer import RecognitionResource
from asr4_streaming.recognizer import add_RecognizerServicer_to_server

from asr4.engines.wav2vec.v1.engine_types import Language

from tests.unit.test_recognizer_service import MockArguments, MockEngine


DEFAULT_ENGLISH_MESSAGE: str = "hello i am up and running received a message from you"


def runServer(serverAddress: str, event: multiprocessing.Event):
    asyncio.run(runServerAsync(serverAddress, event))


def initializeEngine(config, language):
    return MockEngine(
        config,
        language,
    )


async def runServerAsync(serverAddress: str, event: multiprocessing.Event):
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=1),
    )
    arguments = MockArguments(language="en-US")
    with patch.object(RecognizerService, "__init__", lambda x, y: None):
        service = RecognizerService("asr4_config.toml")
        service.logger = logging.getLogger("ASR4")
        service._languageCode = "en-US"
        service._language = Language.EN_US
        service._engine = initializeEngine(arguments.config, arguments.language)
        add_RecognizerServicer_to_server(service, server)
        server.add_insecure_port(serverAddress)
        await server.start()
        event.set()
        await server.wait_for_termination()


@pytest.mark.usefixtures("datadir")
class TestRecognizerService(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def rootdir(self, pytestconfig):
        self.rootdir = str(pytestconfig.rootdir)

    @pytest.fixture(autouse=True)
    def datadir(self, pytestconfig):
        self.datadir = f"{pytestconfig.rootdir}/tests/functional/data"

    @classmethod
    def setUpClass(cls):
        event = multiprocessing.Event()
        cls._serverAddress = "localhost:50060"
        cls._worker = multiprocessing.Process(
            target=runServer, args=(cls._serverAddress, event)
        )
        cls._worker.start()
        event.wait(timeout=180)

    def testRecognizeStreamingRequestOneAudioEnUs(self):
        def _streamingRecognize():
            yield StreamingRecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
            )
            yield StreamingRecognizeRequest(
                audio=b"0000",
            )

        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        response_iterator = RecognizerStub(channel).StreamingRecognize(
            _streamingRecognize(), timeout=10
        )

        for response in response_iterator:
            self.assertEqual(
                response.results.alternatives[0].transcript,
                DEFAULT_ENGLISH_MESSAGE,
            )
        self.assertEqual(
            response.results.is_final,
            True,
        )
        self.assertTrue(0.0 <= response.results.alternatives[0].confidence <= 1.0)

    def testRecognizeStreamingRequestMoreThanOneAudioEnUs(self):
        def _streamingRecognize():
            yield StreamingRecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
            )
            yield StreamingRecognizeRequest(
                audio=b"0000",
            )
            yield StreamingRecognizeRequest(
                audio=b"0000",
            )
            yield StreamingRecognizeRequest(
                audio=b"0000",
            )

        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        response_iterator = RecognizerStub(channel).StreamingRecognize(
            _streamingRecognize(), timeout=10
        )
        for response in response_iterator:
            self.assertEqual(
                response.results.alternatives[0].transcript,
                DEFAULT_ENGLISH_MESSAGE,
            )

        self.assertEqual(
            response.results.is_final,
            True,
        )

        self.assertTrue(0.0 <= response.results.alternatives[0].confidence <= 1.0)

    def testCheckDurationInStreaming(self):
        def _streamingRecognize():
            yield StreamingRecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
            )
            yield StreamingRecognizeRequest(
                audio=b"01234567890123456789",
            )
            yield StreamingRecognizeRequest(
                audio=b"01234567890123456789",
            )
            yield StreamingRecognizeRequest(
                audio=b"0123456789",
            )

        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        response_iterator = RecognizerStub(channel).StreamingRecognize(
            _streamingRecognize(), timeout=10
        )

        for response in response_iterator:
            self.assertEqual(
                response.results.duration.nanos,
                5 * 312400,
            )
            self.assertEqual(
                response.results.duration.seconds,
                0,
            )

    def testCheckDurationInStreamingWithRealAudio8K(self):
        _AUDIO = os.path.join(
            self.datadir, "0e4b2dbd-95c4-4070-ae6d-e79236e73afb_cut_1-channel.wav"
        )

        def _streamingRecognize():
            with wave.open(_AUDIO, "rb") as f:
                yield StreamingRecognizeRequest(
                    config=RecognitionConfig(
                        parameters=RecognitionParameters(
                            language="en-US", sample_rate_hz=f.getframerate()
                        ),
                        resource=RecognitionResource(topic="GENERIC"),
                    ),
                )

                fiveSeconds = 5 * f.getframerate()
                totalRead = 0
                while totalRead < frames:
                    audio = f.readframes(fiveSeconds)
                    yield StreamingRecognizeRequest(audio=audio)
                    totalRead += fiveSeconds

        with wave.open(_AUDIO, "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            (fraction, seconds) = math.modf(frames / float(rate))

        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        response_iterator = RecognizerStub(channel).StreamingRecognize(
            _streamingRecognize(), timeout=10
        )

        for response in response_iterator:
            self.assertEqual(
                response.results.duration.nanos,
                int(fraction * 1_000_000_000),
            )
            self.assertEqual(
                response.results.duration.seconds,
                int(seconds),
            )
            self.assertEqual(
                response.results.end_time.nanos,
                int(fraction * 1_000_000_000),
            )
            self.assertEqual(
                response.results.end_time.seconds,
                int(seconds),
            )

    def testCheckDurationInStreamingWithRealAudio8KWrongSampleRate(self):
        _AUDIO = os.path.join(
            self.datadir, "0e4b2dbd-95c4-4070-ae6d-e79236e73afb_cut_1-channel.wav"
        )

        def _streamingRecognize():
            with wave.open(_AUDIO, "rb") as f:
                yield StreamingRecognizeRequest(
                    config=RecognitionConfig(
                        parameters=RecognitionParameters(
                            language="en-US", sample_rate_hz=16000
                        ),
                        resource=RecognitionResource(topic="GENERIC"),
                    ),
                )

                fiveSeconds = 5 * f.getframerate()
                totalRead = 0
                while totalRead < frames:
                    audio = f.readframes(fiveSeconds)
                    yield StreamingRecognizeRequest(audio=audio)
                    totalRead += fiveSeconds

        with wave.open(_AUDIO, "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            (fraction, seconds) = math.modf(frames / float(rate) / 2)

        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        response_iterator = RecognizerStub(channel).StreamingRecognize(
            _streamingRecognize(), timeout=10
        )

        for response in response_iterator:
            self.assertEqual(
                response.results.duration.nanos,
                int(fraction * 1_000_000_000),
            )
            self.assertEqual(
                response.results.duration.seconds,
                int(seconds),
            )
            self.assertEqual(
                response.results.end_time.nanos,
                int(fraction * 1_000_000_000),
            )
            self.assertEqual(
                response.results.end_time.seconds,
                int(seconds),
            )

    def testCheckDurationInStreamingWithRealAudio16K(self):
        _AUDIO = os.path.join(
            self.datadir, "0e4b2dbd-95c4-4070-ae6d-e79236e73afb_cut_1-channel-16k.wav"
        )

        def _streamingRecognize():
            with wave.open(_AUDIO, "rb") as f:
                yield StreamingRecognizeRequest(
                    config=RecognitionConfig(
                        parameters=RecognitionParameters(
                            language="en-US", sample_rate_hz=f.getframerate()
                        ),
                        resource=RecognitionResource(topic="GENERIC"),
                    ),
                )

                fiveSeconds = 5 * f.getframerate()
                totalRead = 0
                while totalRead < frames:
                    audio = f.readframes(fiveSeconds)
                    yield StreamingRecognizeRequest(audio=audio)
                    totalRead += fiveSeconds

        with wave.open(_AUDIO, "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            (fraction, seconds) = math.modf(frames / float(rate))

        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        response_iterator = RecognizerStub(channel).StreamingRecognize(
            _streamingRecognize(), timeout=10
        )

        for response in response_iterator:
            self.assertEqual(
                response.results.duration.nanos,
                int(fraction * 1_000_000_000),
            )
            self.assertEqual(
                response.results.duration.seconds,
                int(seconds),
            )
            self.assertEqual(
                response.results.end_time.nanos,
                int(fraction * 1_000_000_000),
            )
            self.assertEqual(
                response.results.end_time.seconds,
                int(seconds),
            )

    @classmethod
    def tearDownClass(cls):
        cls._worker.kill()
