import os

import argparse
import multiprocessing
import toml

from asr4.recognizer import Language
from asr4.recognizer import LoggerService
from asr4.recognizer import Server, ServerConfiguration
from asr4.recognizer import DecodingType


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = fixNumberOfJobs(_parseArguments())
    args = TomlConfigurationOverride(args)
    args = SystemVarsOverride(args)
    checkArgsRequired(args)
    logService = LoggerService(args.verbose)
    logService.configureGlobalLogger()
    serve(ServerConfiguration(args), logService)
    logService.stop()


def serve(
        configuration,
        loggerService: LoggerService,
) -> None:
    logger = loggerService.getLogger()
    servers = []
    for i in range(configuration.numberOfServers):
        logger.info("Starting server %s" % i)
        server = Server(configuration, loggerService)
        server.spawn()
        servers.append(server)

    for server in servers:
        server.join()


def _parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python ASR4 Server")
    parser.add_argument(
        "-m",
        "--model-path",
        dest="model",
        help="Path to the model file.",
    )
    parser.add_argument(
        "-d",
        "--dictionary-path",
        dest="vocabulary",
        help="Path to the model's dictionary file, containing all the possible outputs from the model.",
    )
    parser.add_argument(
        "-l",
        "--language",
        dest="language",
        choices=[l.value.lower() for l in Language],
        type=str.lower,
        help="Language of the recognizer service.",
    )
    parser.add_argument(
        "-f",
        "--formatter-model-path",
        dest="formatter",
        help="Path to the formatter model file.",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        dest="gpu",
        action="store_true",
        help="Whether to use GPU instead of CPU",
    )
    parser.add_argument(
        "--host",
        dest="bindAddress",
        help="Hostname address to bind the server to.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        dest="jobs",
        help="Deprecated. Just for backcompatibility issues. Overrides -S, -L and -w and has the same effect as:  -S 1 -L {jobs} -w 0",
    )
    parser.add_argument(
        "-s",
        "--servers",
        type=int,
        dest="servers",
        help="The number of inference servers to be run. Each server will load a whole new inference system.",
    )
    parser.add_argument(
        "-L",
        "--listeners",
        type=int,
        dest="listeners",
        help="Number of gRPC listeners that a server will load. All listeners share the same inference server",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        dest="workers",
        help="The number of workers that a single listener can use to resolve a single request.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=str,
        choices=LoggerService.getLogLevelOptions(),
        help="Log levels. By default reads env variable LOG_LEVEL.",
    )
    parser.add_argument(
        "-D",
        "--decoding-type",
        type=str,
        dest="decoding_type",
        choices=DecodingType._member_names_,
        help="Perform Decoding for each chunk (Local) or for all chunks (Global)",
    )
    parser.add_argument(
        "--lm-algorithm",
        type=str,
        dest="lm_algorithm",
        choices=["viterbi", "kenlm"],
        help="Type of algorithm for language model decoding.",
    )
    parser.add_argument(
        "--lm-lexicon",
        type=str,
        dest="lexicon",
        help="Lexicon for language model.",
    )
    parser.add_argument(
        "--lm-model",
        type=str,
        dest="lm_model",
        help="Path to the language model file.",
    )
    parser.add_argument(
        "-C", "--config", dest="config", help="Path to the asr4 config file"
    )
    return parser.parse_args()

class TestParseArguments(unittest.TestCase):
    def test_parseArguments_with_minimum_args(self):
        # Test with minimum arguments
        args = _parseArguments()
        self.assertIsInstance(args, argparse.Namespace)
        self.assertIsNone(args.model)
        self.assertIsNone(args.vocabulary)
        self.assertIsNone(args.language)
        self.assertIsNone(args.formatter)
        self.assertFalse(args.gpu)
        self.assertIsNone(args.bindAddress)
        self.assertIsNone(args.jobs)
        self.assertIsNone(args.servers)
        self.assertIsNone(args.listeners)
        self.assertIsNone(args.workers)
        self.assertIsNone(args.verbose)
        self.assertIsNone(args.decoding_type)
        self.assertIsNone(args.lm_algorithm)
        self.assertIsNone(args.lexicon)
        self.assertIsNone(args.lm_model)
        self.assertIsNone(args.config)

    def test_parseArguments_with_all_args(self):
        # Test with all arguments set
        args = _parseArguments(
            "-m", "/path/to/model",
            "-d", "/path/to/dictionary",
            "-l", "en",
            "-f", "/path/to/formatter",
            "-g",
            "--host", "localhost",
            "-j", 2,
            "-s", 3,
            "-L", 4,
            "-w", 5,
            "-v", "debug",
            "-D", "local",
            "--lm-algorithm", "viterbi",
            "--lm-lexicon", "/path/to/lexicon",
            "--lm-model", "/path/to/lm_model",
            "-C", "/path/to/config"
        )
        self.assertIsInstance(args, Namespace)
        self.assertEqual(args.model, "/path/to/model")
        self.assertEqual(args.vocabulary, "/path/to/dictionary")
        self.assertEqual(args.language, "en")
        self.assertEqual(args.formatter, "/path/to/formatter")
        self.assertTrue(args.gpu)
        self.assertEqual(args.bindAddress, "localhost")
        self.assertEqual(args.jobs, 2)
        self.assertEqual(args.servers, 3)
        self.assertEqual(args.listeners, 4)
        self.assertEqual(args.workers, 5)
        self.assertEqual(args.verbose, "debug")
        self.assertEqual(args.decoding_type, "local")
        self.assertEqual(args.lm_algorithm, "viterbi")
        self.assertEqual(args.lexicon, "/path/to/lexicon")
        self.assertEqual(args.lm_model, "/path/to/lm_model")
        self.assertEqual(args.config, "/path/to/config")

    def test_parseArguments_with_invalid_args(self):
        # Test with invalid arguments
        with self.assertRaises(SystemExit):
            _ = _parseArguments("-m", "/path/to/model", "-l", "invalid_language")

        with self.assertRaises(SystemExit):
            _ = _parseArguments("-D", "invalid_decoding_type")

def setDefaultBindAddress(args, config):
    config['global'].setdefault('host', '[::]')
    config['global'].setdefault('port', 50051)
    args.bindAddress = config['global']['host'] + ':' + str(config['global']['port'])
    del config['global']['host']
    del config['global']['port']


def TomlConfigurationOverride(args: argparse.Namespace) -> argparse.Namespace:
    config_file = args.config or "asr4_config.toml"
    if os.path.exists(config_file):
        config = toml.load(config_file)

        setDefaultBindAddress(args, config)

        for k, v in config['global'].items():
            setattr(args, k, v)

        args.language = args.language or Language.EN_US.value

        if args.language.lower() in config:
            for k, v in config[args.language.lower()].items():
                setattr(args, k, v)

    return args
def SystemVarsOverride(args: argparse.Namespace) -> argparse.Namespace:
    args.verbose = args.verbose or os.environ.get("LOG_LEVEL", LoggerService.getDefaultLogLevel())
    args.gpu = args.gpu or bool(os.environ.get("ASR4_GPU", False))
    args.servers = args.servers or int(os.environ.get("ASR4_SERVERS", 1))
    args.listeners = args.listeners or int(os.environ.get("ASR4_LISTENERS", 1))
    args.workers = args.workers or int(os.environ.get("ASR4_WORKERS", 2))
    args.decoding_type = args.decoding_type or os.environ.get("ASR4_DECODING_TYPE", "GLOBAL")
    args.lm_algorithm = args.lm_algorithm or os.environ.get("ASR4_LM_ALGORITHM", "viterbi")

    return args

def checkArgsRequired(args):

    if not args.model_path:
        pass

    if args.lm_algorithm and not ( args.lm_model or args.lexicon ):
        pass

    pass


def fixNumberOfJobs(args):
    if args.jobs is not None:
        args.servers = 1
        args.workers = 0
        args.listeners = args.jobs
    return args


if __name__ == "__main__":
    main()
