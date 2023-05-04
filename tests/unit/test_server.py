import multiprocessing
import unittest
import tempfile
from unittest.mock import patch

import argparse, os
from asr4.recognizer import Server, ServerConfiguration
from asr4.recognizer import RecognitionServiceConfiguration
from asr4.recognizer import LoggerService
from bin.server import Asr4ArgParser


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
        self.lexicon = None
        self.lm_model = None
        self.lm_algorithm = "viterbi"
        self.lm_weight = None
        self.word_score = None
        self.sil_score = None
        self.overlap = None
        self.subwords = None


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


class ArgumentParserTests(unittest.TestCase):
    def test_parseArguments(self):
        # Test case 1: Test parsing arguments with some input
        argv = [
            "-m",
            "model.pth",
            "-d",
            "dictionary.txt",
            "-l",
            "en-us",
            "-f",
            "formatter.pth",
            "-g",
            "--host",
            "[::]:50052",
            "-j",
            "4",
            "-s",
            "2",
            "-L",
            "4",
            "-w",
            "8",
            "-v",
            "DEBUG",
            "-D",
            "GLOBAL",
            "--lm-algorithm",
            "kenlm",
            "--lm-lexicon",
            "lexicon.txt",
            "--lm-model",
            "lm_model.pth",
            "-C",
            "config.toml",
            "--lm_weight",
            "0.5",
            "--word_score",
            "0.2",
            "--sil_score",
            "0.1",
        ]
        args = Asr4ArgParser.parseArguments(argv)
        self.assertIsInstance(args, argparse.Namespace)
        self.assertEqual(args.model, "model.pth")
        self.assertEqual(args.vocabulary, "dictionary.txt")
        self.assertEqual(args.language, "en-us")
        self.assertEqual(args.formatter, "formatter.pth")
        self.assertEqual(args.gpu, True)
        self.assertEqual(args.bindAddress, "[::]:50052")
        self.assertEqual(args.jobs, 4)
        self.assertEqual(args.servers, 2)
        self.assertEqual(args.listeners, 4)
        self.assertEqual(args.workers, 8)
        self.assertEqual(args.verbose, "DEBUG")
        self.assertEqual(args.decoding_type, "GLOBAL")
        self.assertEqual(args.lm_algorithm, "kenlm")
        self.assertEqual(args.lexicon, "lexicon.txt")
        self.assertEqual(args.lm_model, "lm_model.pth")
        self.assertEqual(args.config, "config.toml")
        self.assertEqual(args.lm_weight, 0.5)
        self.assertEqual(args.word_score, 0.2)
        self.assertEqual(args.sil_score, 0.1)


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
        # Create a sample TOML config string for testing
        config_str = """
        [global]
        listeners = 2
        workers = 1
        verbose = "DEBUG"
        host = "localhost"
        port = 8080
        gpu = 0

        [en-us]
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
        args.decoding_type = None
        args.lm_algorithm = None

        args = Asr4ArgParser.replaceUndefinedWithConfigFile(args)
        self.assertEqual(args.listeners, 2)
        self.assertEqual(args.workers, 1)
        self.assertEqual(args.verbose, "DEBUG")
        self.assertEqual(args.bindAddress, "localhost:8080")
        self.assertEqual(args.gpu, 0)
        self.assertEqual(args.cpu_version, "2.0.0")
        self.assertEqual(args.gpu_version, "2.0.0")
        self.assertEqual(args.lm_version, "2.0.0")
        self.assertEqual(args.lm_weight, "0.5")
        self.assertEqual(args.word_score, "-0.1")
        self.assertEqual(args.sil_score, "0.2")
        self.assertEqual(args.formatter, "format-model.en-us-2.0.0.fm")


class SystemVarsOverrideTests(unittest.TestCase):
    def test_system_vars_override_with_all_args_set(self):
        args = argparse.Namespace()
        args.verbose = "DEBUG"
        args.gpu = False
        args.bindAddress = "[::]:50052"
        args.servers = 3
        args.listeners = 5
        args.workers = 4
        args.decoding_type = "LOCAL"
        args.lm_algorithm = "beam_search"
        args.lm_weight = 0.5
        args.word_score = -0.2
        args.sil_score = 0.1
        args.config = None
        args.language = None
        args.overlap = 0
        args.subwords = False

        args = Asr4ArgParser.replaceUndefinedWithEnvVariables(args)
        args = Asr4ArgParser.replaceUndefinedWithConfigFile(args)
        args = Asr4ArgParser.replaceUndefinedWithDefaultValues(args)

        # Assert that the values in args have not been changed since they are already set
        self.assertEqual(args.verbose, "DEBUG")
        self.assertEqual(args.gpu, False)
        self.assertEqual(args.bindAddress, "[::]:50052")
        self.assertEqual(args.servers, 3)
        self.assertEqual(args.listeners, 5)
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.decoding_type, "LOCAL")
        self.assertEqual(args.lm_algorithm, "beam_search")
        self.assertEqual(args.lm_weight, 0.5)
        self.assertEqual(args.word_score, -0.2)
        self.assertEqual(args.sil_score, 0.1)
        self.assertEqual(args.overlap, 0)
        self.assertEqual(args.subwords, False)

    def test_system_vars_override_with_env_vars_set(self):
        args = argparse.Namespace()
        args.verbose = None
        args.gpu = None
        args.bindAddress = None
        args.servers = None
        args.listeners = None
        args.workers = None
        args.decoding_type = None
        args.lm_algorithm = None
        args.lm_weight = None
        args.word_score = None
        args.sil_score = None
        args.config = None
        args.language = None
        args.overlap = None
        args.subwords = None

        # Set environment variables for testing
        os.environ["LOG_LEVEL"] = "INFO"
        os.environ["ASR4_GPU"] = "1"
        os.environ["ASR4_HOST"] = "[::]"
        os.environ["ASR4_PORT"] = "50052"
        os.environ["ASR4_SERVERS"] = "2"
        os.environ["ASR4_LISTENERS"] = "3"
        os.environ["ASR4_WORKERS"] = "4"
        os.environ["ASR4_DECODING_TYPE"] = "GLOBAL"
        os.environ["ASR4_LM_ALGORITHM"] = "viterbi"
        os.environ["ASR4_LM_WEIGHT"] = "0.7"
        os.environ["ASR4_WORD_SCORE"] = "-0.5"
        os.environ["ASR4_SIL_SCORE"] = "0.3"
        os.environ["ASR4_OVERLAP"] = "80000"
        os.environ["ASR4_SUBWORDS"] = "1"

        args = Asr4ArgParser.replaceUndefinedWithEnvVariables(args)
        args = Asr4ArgParser.replaceUndefinedWithConfigFile(args)

        # Assert that the values in args have been overridden by the environment variables
        self.assertEqual(args.verbose, "INFO")
        self.assertEqual(args.gpu, "1")
        self.assertEqual(args.bindAddress, "[::]:50052")
        self.assertEqual(args.servers, "2")
        self.assertEqual(args.listeners, "3")
        self.assertEqual(args.workers, "4")
        self.assertEqual(args.decoding_type, "GLOBAL")
        self.assertEqual(args.lm_algorithm, "viterbi")
        self.assertEqual(args.lm_weight, "0.7")
        self.assertEqual(args.word_score, "-0.5")
        self.assertEqual(args.sil_score, "0.3")
        self.assertEqual(args.overlap, "80000")
        self.assertEqual(args.subwords, "1")

        # Clean up environment variables after the test
        del os.environ["LOG_LEVEL"]
        del os.environ["ASR4_GPU"]
        del os.environ["ASR4_HOST"]
        del os.environ["ASR4_PORT"]
        del os.environ["ASR4_SERVERS"]
        del os.environ["ASR4_LISTENERS"]
        del os.environ["ASR4_WORKERS"]
        del os.environ["ASR4_DECODING_TYPE"]
        del os.environ["ASR4_LM_ALGORITHM"]
        del os.environ["ASR4_LM_WEIGHT"]
        del os.environ["ASR4_WORD_SCORE"]
        del os.environ["ASR4_SIL_SCORE"]
        del os.environ["ASR4_SUBWORDS"]


class DefaultValuesTests(unittest.TestCase):
    def test_system_vars_override_with_defaults(self):
        args = argparse.Namespace()

        args.bindAddress = "[::]:50052"
        args.decoding_type = "LOCAL"
        args.lm_algorithm = "beam_search"
        args.lm_weight = "0.5"
        args.servers = "3"
        args.verbose = "DEBUG"
        args.word_score = "-0.2"
        args.config = None
        args.gpu = None
        args.language = None
        args.listeners = None
        args.sil_score = None
        args.workers = None
        args.overlap = None
        args.subwords = "1"

        args = Asr4ArgParser.replaceUndefinedWithDefaultValues(args)

        # Assert that the values in args are transformed to correct types
        self.assertEqual(args.verbose, "DEBUG")
        self.assertEqual(args.gpu, False)
        self.assertEqual(args.bindAddress, "[::]:50052")
        self.assertEqual(args.servers, 3)
        self.assertEqual(args.listeners, 1)
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.decoding_type, "LOCAL")
        self.assertEqual(args.lm_algorithm, "beam_search")
        self.assertEqual(args.lm_weight, 0.5)
        self.assertEqual(args.word_score, -0.2)
        self.assertEqual(args.sil_score, 0.0)
        self.assertEqual(args.subwords, True)


class TestCheckArgsRequired(unittest.TestCase):
    def setUp(self):
        # Create a mock argparse.Namespace object with required attributes
        self.args = argparse.Namespace()
        self.args.model = None
        self.args.vocabulary = None
        self.args.language = "en-us"
        self.args.lm_algorithm = "kenlm"
        self.args.lm_model = None
        self.args.lexicon = None
        self.args.gpu = False
        self.args.gpu_version = "1.2.0"
        self.args.cpu_version = "1.1.0"
        self.args.lm_version = "1.3.0"

    def test_checkArgsRequired_with_model_specified(self):
        # Test when args.model is already specified, no exception should be raised
        self.args.model = "model.onnx"
        with self.assertRaises(ValueError):
            Asr4ArgParser.checkArgsRequired(self.args)
        self.args.model = None
        self.args.vocabulary = "model.dict.ltr.txt"
        with self.assertRaises(ValueError):
            Asr4ArgParser.checkArgsRequired(self.args)

    @patch("os.path.exists")
    def test_checkArgsRequired_with_standard_model_paths(self, mock_exists):
        # Test when standard model paths exist, args.model and args.vocabulary should be updated
        mock_exists.side_effect = [True, True, True, True]
        result = Asr4ArgParser.checkArgsRequired(self.args)
        self.assertEqual(result.model, "asr4-en-us.onnx")
        self.assertEqual(result.vocabulary, "dict.ltr.txt")

    @patch("os.path.exists")
    def test_checkArgsRequired_with_gpu_model_paths(self, mock_exists):
        # Test when GPU model paths exist and args.gpu is True, args.model and args.vocabulary should be updated
        mock_exists.side_effect = [False, True, True, True, True]
        self.args.gpu = True
        self.args.gpu_version = "1.2.0"
        result = Asr4ArgParser.checkArgsRequired(self.args)
        self.assertEqual(result.model, "asr4-en-us-1.2.0.onnx")
        self.assertEqual(result.vocabulary, "asr4-en-us-1.2.0.dict.ltr.txt")

    @patch("os.path.exists")
    def test_checkArgsRequired_with_cpu_model_paths(self, mock_exists):
        # Test when CPU model paths exist and args.gpu is False, args.model and args.vocabulary should be updated
        mock_exists.side_effect = [False, True, True, True, True]
        self.args.gpu = False
        self.args.cpu_version = "1.1.0"
        result = Asr4ArgParser.checkArgsRequired(self.args)
        self.assertEqual(result.model, "asr4-en-us-1.1.0.onnx")
        self.assertEqual(result.vocabulary, "asr4-en-us-1.1.0.dict.ltr.txt")

    @patch("os.path.exists")
    def test_checkArgsRequired_with_lm_algorithm_kenlm_and_specified_lm_model_lexicon(
        self, mock_exists
    ):
        # Create a mock argparse.Namespace object
        self.args.lm_algorithm = "kenlm"
        self.args.lm_model = "lm_model_path"
        self.args.lexicon = "lm_lexicon_path"
        # Mock the os.path.exists function to return True for the specified paths
        mock_exists.side_effect = [True, True, True, True]

        result = Asr4ArgParser.checkArgsRequired(self.args)
        self.assertEqual(result, self.args)

    @patch("os.path.exists")
    def test_checkArgsRequired_with_lm_algorithm_kenlm_and_no_lm_model_lexicon(
        self, mock_exists
    ):
        # Create a mock argparse.Namespace object
        self.args.lm_algorithm = "kenlm"
        # Mock the os.path.exists function to return False for all paths
        mock_exists.return_value = False

        with self.assertRaises(ValueError):
            Asr4ArgParser.checkArgsRequired(self.args)


class TestServer(unittest.TestCase):
    def testServerNoSpawn(self):
        # It is not possible to test Server because it has not a stop() function
        multiprocessing.set_start_method("spawn", force=True)
        loggerService = LoggerService("DEBUG")
        server = Server(ServerConfiguration(MockArguments()), loggerService)
        server.join()
