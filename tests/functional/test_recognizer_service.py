import os
import threading

import grpc
import math
import wave
import pytest
import asyncio
import unittest
import multiprocessing
from concurrent import futures

from asr4.recognizer import Language
from asr4.recognizer import RecognizerStub
from asr4.recognizer import RecognizerService, RecognitionServiceConfiguration
from asr4.recognizer import RecognizeRequest
from asr4.recognizer import StreamingRecognizeRequest
from asr4.recognizer import RecognitionConfig
from asr4.recognizer import RecognitionParameters
from asr4.recognizer import RecognitionResource
from asr4.recognizer import add_RecognizerServicer_to_server
from asr4.recognizer import LoggerService

from tests.unit.test_recognizer_service import (
    MockOnnxSession,
    MockArguments,
    MockRecognitionServiceConfiguration,
)
from tests.unit.test_onnx_runtime import MockFormatter

DEFAULT_ENGLISH_MESSAGE: str = "hello i am up and running received a message from you"


def runServer(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def runServerAsync(
    server: grpc.aio.Server, serverAddress: str, event: multiprocessing.Event
):
    configuration = MockRecognitionServiceConfiguration(MockArguments())
    configuration.language = Language.EN_US
    configuration.vocabulary = None
    configuration.local_formatting = True
    add_RecognizerServicer_to_server(RecognizerService(configuration), server)
    server.add_insecure_port(serverAddress)
    print("HERE1")
    await server.start()
    print("HERE2")
    event.set()
    print("HERE3")
    await server.wait_for_termination()
    print("HERE4")


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
        loop = asyncio.new_event_loop()
        loggerService = LoggerService("TRACE")
        logsQueue = loggerService.getQueue()
        logsQueue.configureGlobalLogger()
        logger = logsQueue.getLogger()
        cls._event = threading.Event()
        cls._shutdown_event = threading.Event()
        cls._serverAddress = "localhost:50060"
        cls._server: grpc.aio.Server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=1),
        )
        cls._worker = threading.Thread(target=runServer, args=(loop,))
        cls._worker.start()
        cls.task = asyncio.run_coroutine_threadsafe(
            runServerAsync(cls._server, cls._serverAddress, cls._event), loop
        )
        cls._event.wait(timeout=180)
        print("SERVER IS UP")

    def testRecognizeRequestEnUs(self):
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(
                    language="en-US", sample_rate_hz=16000
                ),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"0000",
        )
        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        response = RecognizerStub(channel).Recognize(request, timeout=10)
        self.assertEqual(
            response.alternatives[0].transcript,
            DEFAULT_ENGLISH_MESSAGE,
        )

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
        self.assertEqual(
            response.results.alternatives[0].confidence,
            0.995789647102356,
        )

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

        self.assertEqual(
            response.results.alternatives[0].confidence,
            0.995789647102356,
        )

    def testRecognizeRequestEs(self):
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="es", sample_rate_hz=16000),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"0000",
        )
        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        with self.assertRaises(grpc._channel._InactiveRpcError):
            RecognizerStub(channel).Recognize(request, timeout=10)

    def testRecognizeRequestPtBr(self):
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(
                    language="pt-BR", sample_rate_hz=16000
                ),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"0000",
        )
        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        with self.assertRaises(grpc._channel._InactiveRpcError):
            RecognizerStub(channel).Recognize(request, timeout=10)

    def testRecognizeStereoAudio(self):
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(
                    language="en-US", sample_rate_hz=16000
                ),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=open(
                os.path.join(
                    self.datadir, "0e4b2dbd-95c4-4070-ae6d-e79236e73afb_cut.wav"
                ),
                "rb",
            ).read(),
        )
        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        response = RecognizerStub(channel).Recognize(request, timeout=10)
        self.assertEqual(
            response.alternatives[0].transcript,
            DEFAULT_ENGLISH_MESSAGE,
        )

    def testRecognizeRequest8kHz(self):
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US", sample_rate_hz=8000),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"0000",
        )
        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        response = RecognizerStub(channel).Recognize(request, timeout=10)
        self.assertEqual(
            response.alternatives[0].transcript,
            DEFAULT_ENGLISH_MESSAGE,
        )

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
        cls._server.stop()


def runServerPartialDecoding(serverAddress: str, event: multiprocessing.Event):
    asyncio.run(runServerAsyncPartialDecoding(serverAddress, event))


async def runServerAsyncPartialDecoding(
    serverAddress: str, event: multiprocessing.Event
):
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=1),
    )
    configuration = MockRecognitionServiceConfiguration(MockArguments())
    configuration.language = Language.EN_US
    configuration.vocabulary = None
    configuration.formatterModelPath = "path_to_formatter/formatter.fm"
    configuration.decodingType = "LOCAL"
    configuration.lmAlgorithm = "kenlm"
    configuration.lmFile = "path_to_lm/lm.bin"
    configuration.lexicon = "path_to_lm/lm.lexicon.txt"
    configuration.local_formatting = "True"
    add_RecognizerServicer_to_server(RecognizerService(configuration), server)
    server.add_insecure_port(serverAddress)
    await server.start()
    event.set()
    await server.wait_for_termination()


@pytest.mark.usefixtures("datadir")
class TestRecognizerServiceOnlineDecoding(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def rootdir(self, pytestconfig):
        self.rootdir = str(pytestconfig.rootdir)

    @pytest.fixture(autouse=True)
    def datadir(self, pytestconfig):
        self.datadir = f"{pytestconfig.rootdir}/tests/functional/data"

    @classmethod
    def setUpClass(cls):
        cls._event = multiprocessing.Event()
        cls._serverAddress = "localhost:50060"
        cls._worker = multiprocessing.Process(
            target=runServerPartialDecoding, args=(cls._serverAddress, cls._event)
        )
        cls._worker.start()
        cls._event.wait(timeout=240)

    def testRecognizeRequest8kHz(self):
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US", sample_rate_hz=8000),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"0000",
        )
        channel = grpc.insecure_channel(
            TestRecognizerServiceOnlineDecoding._serverAddress
        )
        response = RecognizerStub(channel).Recognize(request, timeout=200)
        self.assertEqual(
            response.alternatives[0].transcript,
            DEFAULT_ENGLISH_MESSAGE,
        )

    @classmethod
    def tearDownClass(cls):
        pass
