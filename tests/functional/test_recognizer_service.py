import unittest

import grpc
import asyncio
import multiprocessing
from concurrent import futures

from asr4.recognizer import RecognizerStub
from asr4.recognizer import RecognizerService
from asr4.recognizer import RecognizeRequest
from asr4.recognizer import RecognitionConfig
from asr4.recognizer import RecognitionParameters
from asr4.recognizer import RecognitionResource
from asr4.recognizer import add_RecognizerServicer_to_server

from tests.unit.test_recognizer_service import MockOnnxSession

DEFAULT_ENGLISH_MESSAGE: str = "hello i am up and running received a message from you"
DEFAULT_SPANISH_MESSAGE: str = (
    "Hola, estoy levantado y en marcha. ¡He recibido un mensaje tuyo!"
)
DEFAULT_PORTUGUESE_MESSAGE: str = "Olá, estou de pé, recebi uma mensagem sua!"


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


class TestRecognizerService(unittest.TestCase):
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
            response.text,
            DEFAULT_ENGLISH_MESSAGE,
        )

    def testRecognizeRequestEsEs(self):
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(
                    language="es-ES", sample_rate_hz=16000
                ),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"0000",
        )
        channel = grpc.insecure_channel(TestRecognizerService._serverAddress)
        response = RecognizerStub(channel).Recognize(request, timeout=10)
        self.assertEqual(
            response.text,
            DEFAULT_SPANISH_MESSAGE,
        )

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
        response = RecognizerStub(channel).Recognize(request, timeout=10)
        self.assertEqual(
            response.text,
            DEFAULT_PORTUGUESE_MESSAGE,
        )

    @classmethod
    def tearDownClass(cls):
        cls._worker.kill()
