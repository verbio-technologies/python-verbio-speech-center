import multiprocessing
import unittest
import tempfile
from unittest.mock import patch

import argparse, os
from asr4_streaming.recognizer import Server, ServerConfiguration
from asr4_streaming.recognizer import Logger
from bin.server import Asr4ArgParser


class MockArguments(argparse.Namespace):
    def __init__(self):
        super().__init__()
        self.config = "config.toml"
        self.bindAddress = "bind:address"
        self.servers = 3
        self.listeners = 2
        self.vocabulary = None
        self.formatter = None
        self.language = "es"
        self.model = "model.onnx"
        self.gpu = False
        self.workers = 4
        self.lexicon = None
        self.lm_model = None
        self.lm_algorithm = "viterbi"
        self.lm_weight = None
        self.word_score = None
        self.sil_score = None
        self.overlap = None
        self.subwords = None
        self.local_formatting = False
        self.maxChunksForDecoding = 1
        self.verbose = "INFO"


class TestServerConfiguration(unittest.TestCase):
    def testConfiguration(self):
        arguments = MockArguments()
        configuration = ServerConfiguration(arguments)
        self.assertEqual(configuration.numberOfListeners, arguments.listeners)
        self.assertEqual(configuration.numberOfServers, arguments.servers)
        self.assertEqual(configuration.bindAddress, arguments.bindAddress)


class ArgumentParserTests(unittest.TestCase):
    def testParseArguments(self):
        argv = ["-v", "DEBUG", "-C", "config.toml"]
        args = Asr4ArgParser.parseArguments(argv)
        self.assertIsInstance(args, argparse.Namespace)
        self.assertEqual(args.verbose, "DEBUG")


class SetDefaultBindAddressTests(unittest.TestCase):
    def test_setDefaultBindAddress(self):
        args = argparse.Namespace()
        args.bindAddress = None
        args.language = ""
        config = {"global": {"host": "[::]", "port": 50052}}
        Asr4ArgParser.fillArgsFromTomlFile(args, config)
        self.assertEqual(args.bindAddress, "[::]:50052")
        self.assertNotIn("host", config["global"])
        self.assertNotIn("port", config["global"])


class TestreplaceUndefinedWithConfigFile(unittest.TestCase):
    def test_toml_configuration_override(self):
        config_str = """
        [global]
        foo = "bar"
        listeners = 2
        workers = 1
        verbose = "DEBUG"
        host = "localhost"
        port = 8080
        gpu = 0
        cpu_version = "2.0.0"
        gpu_version = "2.0.0"
        lm_version = "2.0.0"
        lm_weight = "0.5"
        word_score = "-0.1"
        sil_score = "0.2"
        formatter = "format-model.en-us-2.0.0.fm"
        """
        tmpfile = tempfile.NamedTemporaryFile(mode="w")
        with open(tmpfile.name, "w") as f:
            f.write(config_str)

        args = argparse.Namespace(delete=True)
        args.language = "en-us"
        args.config = tmpfile.name
        args.bindAddress = None
        args.servers = None

        args = Asr4ArgParser.replaceUndefinedWithConfigFile(args)
        self.assertEqual(args.listeners, 2)
        self.assertEqual(args.workers, 1)
        self.assertEqual(args.verbose, "DEBUG")
        self.assertEqual(args.bindAddress, "localhost:8080")


class SystemVarsOverrideTests(unittest.TestCase):
    def test_system_vars_override_with_all_args_set(self):
        args = argparse.Namespace()
        args.verbose = "DEBUG"
        args.bindAddress = "[::]:50052"
        args.servers = 3
        args.listeners = 5
        args.workers = 4
        args.config = None

        args = Asr4ArgParser.replaceUndefinedWithEnvVariables(args)
        args = Asr4ArgParser.replaceUndefinedWithConfigFile(args)
        args = Asr4ArgParser.replaceUndefinedWithDefaultValues(args)

        self.assertEqual(args.verbose, "DEBUG")
        self.assertEqual(args.bindAddress, "[::]:50052")
        self.assertEqual(args.servers, 3)
        self.assertEqual(args.listeners, 5)
        self.assertEqual(args.workers, 4)

    def testSystemVarsOverrideWithEnvVarsSet(self):
        args = argparse.Namespace()
        args.verbose = None
        args.config = None
        os.environ["LOG_LEVEL"] = "INFO"
        args = Asr4ArgParser.replaceUndefinedWithEnvVariables(args)
        args = Asr4ArgParser.replaceUndefinedWithConfigFile(args)
        del os.environ["LOG_LEVEL"]
        self.assertEqual(args.verbose, "INFO")


class DefaultValuesTests(unittest.TestCase):
    def test_system_vars_override_with_defaults(self):
        args = argparse.Namespace()
        args.bindAddress = "[::]:50052"
        args.servers = "3"
        args.verbose = "DEBUG"
        args = Asr4ArgParser.replaceUndefinedWithDefaultValues(args)
        self.assertEqual(args.verbose, "DEBUG")
        self.assertEqual(args.bindAddress, "[::]:50052")
        self.assertEqual(args.servers, 3)
        self.assertEqual(args.listeners, 1)
        self.assertEqual(args.workers, 2)


class TestServer(unittest.TestCase):
    def testServerNoSpawn(self):
        # It is not possible to test Server because it has not a stop() function
        multiprocessing.set_start_method("spawn", force=True)
        server = Server(ServerConfiguration(MockArguments()))
        server.join()
