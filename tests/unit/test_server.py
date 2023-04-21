import multiprocessing
import unittest
from unittest.mock import patch
import argparse
from asr4.recognizer import Server, ServerConfiguration
from asr4.recognizer import RecognitionServiceConfiguration
from asr4.recognizer import LoggerService
from asr4.recognizer import Language
from bin.server import (
    setDefaultBindAddress,
    TomlConfigurationOverride,
    SystemVarsOverride,
    checkArgsRequired,
)


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


class TestSetDefaultBindAddress(unittest.TestCase):
    def test_setDefaultBindAddress(self):
        args = argparse.Namespace()
        config = {"global": {}}

        setDefaultBindAddress(args, config)

        self.assertEqual(args.bindAddress, "[::]:50051")
        self.assertNotIn("host", config["global"])
        self.assertNotIn("port", config["global"])

    def test_setDefaultBindAddress_with_existing_host_port(self):
        args = argparse.Namespace()
        config = {"global": {"host": "127.0.0.1", "port": 8080}}

        setDefaultBindAddress(args, config)

        self.assertEqual(args.bindAddress, "127.0.0.1:8080")
        self.assertNotIn("host", config["global"])
        self.assertNotIn("port", config["global"])


class TestTomlConfigurationOverride(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace()

    @patch("os.path.exists")
    @patch("toml.load")
    def test_TomlConfigurationOverride(self, mock_toml_load, mock_os_path_exists):
        mock_os_path_exists.return_value = True
        mock_toml_load.return_value = {
            "global": {"host": "127.0.0.1", "port": 8080},
            "en_us": {"language": "en_us", "setting1": "value1"},
        }

        args = TomlConfigurationOverride(self.args)

        self.assertEqual(args.bindAddress, "127.0.0.1:8080")
        self.assertEqual(args.language, "en_us")
        self.assertEqual(args.setting1, "value1")

        mock_os_path_exists.assert_called_once_with(
            self.args.config or "asr4_config.toml"
        )
        mock_toml_load.assert_called_once_with(self.args.config or "asr4_config.toml")

    @patch("os.path.exists")
    @patch("toml.load")
    def test_TomlConfigurationOverride_with_default_config_file(
        self, mock_toml_load, mock_os_path_exists
    ):
        mock_os_path_exists.return_value = False

        args = TomlConfigurationOverride(self.args)

        self.assertIsNone(args.bindAddress)
        self.assertEqual(args.language, Language.EN_US.value)

        mock_os_path_exists.assert_called_once_with(
            self.args.config or "asr4_config.toml"
        )
        mock_toml_load.assert_not_called()


class TestSystemVarsOverride(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace()

    @patch("os.environ")
    def test_SystemVarsOverride(self, mock_os_environ):
        mock_os_environ.get.side_effect = lambda key, default=None: {
            "LOG_LEVEL": "debug",
            "ASR4_GPU": "true",
            "ASR4_SERVERS": "3",
            "ASR4_LISTENERS": "2",
            "ASR4_WORKERS": "4",
            "ASR4_DECODING_TYPE": "CTC",
            "ASR4_LM_ALGORITHM": "beam_search",
        }.get(key, default)

        args = SystemVarsOverride(self.args)

        self.assertEqual(args.verbose, "debug")
        self.assertEqual(args.gpu, True)
        self.assertEqual(args.servers, 3)
        self.assertEqual(args.listeners, 2)
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.decoding_type, "CTC")
        self.assertEqual(args.lm_algorithm, "beam_search")

        mock_os_environ.get.assert_has_calls(
            [
                unittest.mock.call("LOG_LEVEL", LoggerService.getDefaultLogLevel()),
                unittest.mock.call("ASR4_GPU", False),
                unittest.mock.call("ASR4_SERVERS", 1),
                unittest.mock.call("ASR4_LISTENERS", 1),
                unittest.mock.call("ASR4_WORKERS", 2),
                unittest.mock.call("ASR4_DECODING_TYPE", "GLOBAL"),
                unittest.mock.call("ASR4_LM_ALGORITHM", "viterbi"),
            ]
        )

    @patch("os.environ")
    def test_SystemVarsOverride_with_defaults(self, mock_os_environ):
        mock_os_environ.get.return_value = None

        args = SystemVarsOverride(self.args)

        self.assertIsNone(args.verbose)
        self.assertIsNone(args.gpu)
        self.assertIsNone(args.servers)
        self.assertIsNone(args.listeners)
        self.assertIsNone(args.workers)
        self.assertIsNone(args.decoding_type)
        self.assertIsNone(args.lm_algorithm)

        mock_os_environ.get.assert_has_calls(
            [
                unittest.mock.call("LOG_LEVEL", LoggerService.getDefaultLogLevel()),
                unittest.mock.call("ASR4_GPU", False),
                unittest.mock.call("ASR4_SERVERS", 1),
                unittest.mock.call("ASR4_LISTENERS", 1),
                unittest.mock.call("ASR4_WORKERS", 2),
                unittest.mock.call("ASR4_DECODING_TYPE", "GLOBAL"),
                unittest.mock.call("ASR4_LM_ALGORITHM", "viterbi"),
            ]
        )

    @patch("os.environ")
    def test_SystemVarsOverride_with_partial_defaults(self, mock_os_environ):
        mock_os_environ.get.side_effect = lambda key, default=None: {
            "LOG_LEVEL": "debug",
            "ASR4_GPU": "true",
            "ASR4_SERVERS": "3",
        }.get(key, default)

        args = SystemVarsOverride(self.args)

        self.assertEqual(args.verbose, "debug")
        self.assertEqual(args.gpu, True)
        self.assertEqual(args.servers, 3)
        self.assertIsNone(args.listeners)
        self.assertIsNone(args.workers)
        self.assertIsNone(args.decoding_type)
        self.assertIsNone(args.lm_algorithm)

        mock_os_environ.get.assert_has_calls(
            [
                unittest.mock.call("LOG_LEVEL", LoggerService.getDefaultLogLevel()),
                unittest.mock.call("ASR4_GPU", False),
                unittest.mock.call("ASR4_SERVERS", 1),
                unittest.mock.call("ASR4_LISTENERS", 1),
                unittest.mock.call("ASR4_WORKERS", 2),
                unittest.mock.call("ASR4_DECODING_TYPE", "GLOBAL"),
                unittest.mock.call("ASR4_LM_ALGORITHM", "viterbi"),
            ]
        )


class TestServer(unittest.TestCase):
    def testServerNoSpawn(self):
        # It is not possible to test Server because it has not a stop() function
        multiprocessing.set_start_method("spawn", force=True)
        loggerService = LoggerService("DEBUG")
        server = Server(ServerConfiguration(MockArguments()), loggerService)
        server.join()
        unittest.main()
