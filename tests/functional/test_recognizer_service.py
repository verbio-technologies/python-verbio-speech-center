import os
import grpc
import pytest
import asyncio
import unittest
import multiprocessing
from concurrent import futures

from asr4.recognizer import RecognizerStub
from asr4.recognizer import RecognizerService
from asr4.recognizer import RecognizeRequest
from asr4.recognizer import StreamingRecognizeRequest
from asr4.recognizer import RecognitionConfig
from asr4.recognizer import RecognitionParameters
from asr4.recognizer import RecognitionResource
from asr4.recognizer import add_RecognizerServicer_to_server

from tests.unit.test_recognizer_service import MockOnnxSession

DEFAULT_ENGLISH_MESSAGE: str = "hello i am up and running received a message from you"


def runServer(serverAddress: str, event: multiprocessing.Event):
    asyncio.run(runServerAsync(serverAddress, event))


async def runServerAsync(serverAddress: str, event: multiprocessing.Event):
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=1),
    )
    add_RecognizerServicer_to_server(RecognizerService(MockOnnxSession("")), server)
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
                response.results.alternatives[0].confidence,
                1.0,
            )
        self.assertEqual(
            response.results.is_final,
            True,
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
                response.results.alternatives[0].confidence,
                1.0,
            )

        self.assertEqual(
            response.results.is_final,
            True,
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

    @classmethod
    def tearDownClass(cls):
        cls._worker.kill()
