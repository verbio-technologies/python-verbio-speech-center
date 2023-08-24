import grpc
import pytest
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

from tests.unit.test_recognizer_service import (
    MockArguments,
    MockRecognitionServiceConfiguration,
)

DEFAULT_ENGLISH_MESSAGE: str = "hello i am up and running received a message from you"


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
        event = multiprocessing.Event()
        cls._serverAddress = "localhost:50060"
        cls._worker = multiprocessing.Process(
            target=runServerPartialDecoding, args=(cls._serverAddress, event)
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

        channel = grpc.insecure_channel(
            TestRecognizerServiceOnlineDecoding._serverAddress
        )
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

        channel = grpc.insecure_channel(
            TestRecognizerServiceOnlineDecoding._serverAddress
        )
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

    @classmethod
    def tearDownClass(cls):
        cls._worker.kill()
