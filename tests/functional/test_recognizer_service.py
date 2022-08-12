from time import sleep
from unittest import IsolatedAsyncioTestCase
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
from asr4.recognizer import RecognizeResponse
from asr4.recognizer import add_RecognizerServicer_to_server


def runServer(serverAddress: str):
    asyncio.run(runServerAsync(serverAddress))

async def runServerAsync(serverAddress: str):
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=1),
    )
    add_RecognizerServicer_to_server(RecognizerService(), server)
    server.add_insecure_port(serverAddress)
    await server.start()
    await server.wait_for_termination()


class TestRecognizerService(unittest.TestCase):
    def setUp(self):
        print("a")
        self._serverAddress = "[::]:50060"
        self._worker = multiprocessing.Process(target=runServer, args=(self._serverAddress,))
        self._worker.start()

        
    def testRecognizeRequestEnUs(self):
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US", sample_rate_hz=16000),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"SOMETHING",
        )
        channel = grpc.insecure_channel(self._serverAddress)
        response = RecognizerStub(channel).Recognize(request, timeout=10)
        self.assertEqual(response.text, "Hello, I am up and running. Received a message from you!")

    
    def testRecognizeRequestEsEs(self):
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="es-ES", sample_rate_hz=16000),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"SOMETHING",
        )
        channel = grpc.insecure_channel(self._serverAddress)
        response = RecognizerStub(channel).Recognize(request, timeout=10)
        self.assertEqual(response.text, "Hola, estoy levantado y en marcha. ¡He recibido un mensaje tuyo!")

    
    def testRecognizeRequestPtBr(self):
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="pt-BR", sample_rate_hz=16000),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"SOMETHING",
        )
        channel = grpc.insecure_channel(self._serverAddress)
        response = RecognizerStub(channel).Recognize(request, timeout=10)
        self.assertEqual(response.text, "Olá, estou de pé, recebi uma mensagem sua!")

    
    def tearDown(self):
        print("b")
        self._worker.kill()