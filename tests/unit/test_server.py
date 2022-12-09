import multiprocessing
import unittest
import argparse
from asr4.recognizer import Server, ServerConfiguration
from asr4.recognizer import RecognitionServiceConfiguration
from asr4.recognizer import LoggerService


class MockArguments(argparse.Namespace):
    def __init__(self):
        super().__init__()
        self.bindAddress = "bind:address"
        self.servers = 3
        self.listeners = 2
        self.vocabulary = None
        self.formatter = None
        self.language = "es"
        self.model = "model.onnx"
        self.gpu = False
        self.workers = 4


class TestServerConfiguration(unittest.TestCase):
    def testConfiguration(self):
        arguments = MockArguments()
        configuration = ServerConfiguration(arguments)
        self.assertEqual(configuration.numberOfListeners, arguments.listeners)
        self.assertEqual(configuration.numberOfServers, arguments.servers)
        self.assertEqual(configuration.bindAddress, arguments.bindAddress)
        self.assertEqual(
            type(configuration.getServiceConfiguration()),
            RecognitionServiceConfiguration,
        )


class TestServer(unittest.TestCase):
    def testServerNoSpawn(self):
        # It is not possible to test Server because it has not a stop() function
        multiprocessing.set_start_method("spawn", force=True)
        loggerService = LoggerService("DEBUG")
        server = Server(ServerConfiguration(MockArguments()), loggerService)
        server.join()
