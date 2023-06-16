import os, sys

import argparse
import multiprocessing
import toml

from asr4.recognizer import Language
from asr4.recognizer import LoggerService
from asr4.recognizer import Server, ServerConfiguration
from asr4.recognizer import DecodingType


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = Asr4ArgParser(sys.argv[1:]).getArgs()
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


class Asr4ArgParser:
    def __init__(self, argv):
        self.argv = argv

    def getArgs(self) -> argparse.Namespace:
        args = Asr4ArgParser.fixNumberOfJobs(Asr4ArgParser.parseArguments(self.argv))
        args = Asr4ArgParser.replaceUndefinedWithEnvVariables(args)
        args = Asr4ArgParser.replaceUndefinedWithConfigFile(args)
        args = Asr4ArgParser.replaceUndefinedWithDefaultValues(args)
        args = Asr4ArgParser.checkArgsRequired(args)
        return args

    def parseArguments(args: list) -> argparse.Namespace:
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
            "-W",
            "--subwords",
            dest="subwords",
            default=False,
            action="store_true",
            help="The final words have to be constructed from word-pieces",
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
        parser.add_argument(
            "--lm_weight",
            dest="lm_weight",
            type=float,
            help="Language Model weight for KenLM",
        )
        parser.add_argument(
            "--word_score",
            dest="word_score",
            type=float,
            help="Word score (penalty) Weight for KenLM",
        )
        parser.add_argument(
            "--sil_score", dest="sil_score", type=float, help="Silence weight for KenLM"
        )
        parser.add_argument(
            "--overlap",
            "-o",
            dest="overlap",
            type=int,
            help="Size of overlapping windows when doing partial decoding",
        )
        parser.add_argument(
            "-local-formatting",
            dest="local_formatting",
            action="store_true",
            help="Perform local formatting when partial decoding",
        )
        parser.add_argument(
            "--maxChunksForDeconding",
            type=int,
            dest="maxChunksForDeconding",
            help="The number of chunks of audio until perform local decoding.",
        )
        return parser.parse_args(args)

    def fixNumberOfJobs(args):
        if args.jobs is not None:
            args.servers = 1
            args.workers = 0
            args.listeners = args.jobs
        return args

    def replaceUndefinedWithEnvVariables(
        args: argparse.Namespace,
    ) -> argparse.Namespace:
        args.verbose = args.verbose or os.environ.get(
            "LOG_LEVEL", LoggerService.getDefaultLogLevel()
        )
        args.gpu = args.gpu or os.environ.get("ASR4_GPU")
        args.servers = args.servers or os.environ.get("ASR4_SERVERS")
        args.listeners = args.listeners or os.environ.get("ASR4_LISTENERS")
        args.workers = args.workers or os.environ.get("ASR4_WORKERS")
        args.decoding_type = args.decoding_type or os.environ.get("ASR4_DECODING_TYPE")
        args.lm_algorithm = args.lm_algorithm or os.environ.get("ASR4_LM_ALGORITHM")
        args.lm_weight = args.lm_weight or os.environ.get("ASR4_LM_WEIGHT")
        args.word_score = args.word_score or os.environ.get("ASR4_WORD_SCORE")
        args.sil_score = args.sil_score or os.environ.get("ASR4_SIL_SCORE")
        args.overlap = args.overlap or os.environ.get("ASR4_OVERLAP")
        args.subwords = args.subwords or os.environ.get("ASR4_SUBWORDS")
        args.local_formatting = args.local_formatting or os.environ.get(
            "ASR4_LOCAL_FORMATTING"
        )
        if os.environ.get("ASR4_HOST") and os.environ.get("ASR4_PORT"):
            args.bindAddress = (
                f"{os.environ.get('ASR4_HOST')}:{os.environ.get('ASR4_PORT')}"
            )
        return args

    def replaceUndefinedWithConfigFile(args: argparse.Namespace) -> argparse.Namespace:
        configFile = args.config or "asr4_config.toml"
        args.language = args.language or Language.EN_US.value
        if os.path.exists(configFile):
            config = toml.load(configFile)
            config.setdefault("global", {})
            args = Asr4ArgParser.fillArgsFromTomlFile(args, config)
        return args

    def fillArgsFromTomlFile(args: argparse.Namespace, config):
        if not args.bindAddress:
            if config["global"].setdefault("host") and config["global"].setdefault(
                "port"
            ):
                args.bindAddress = (
                    f"{config['global']['host']}:{config['global']['port']}"
                )
                del config["global"]["host"]
                del config["global"]["port"]
        for k, v in config["global"].items():
            setattr(args, k, getattr(args, k, None) or v)
        if args.language.lower() in config:
            for k, v in config[args.language.lower()].items():
                setattr(args, k, getattr(args, k, None) or v)
        return args

    def replaceUndefinedWithDefaultValues(
        args: argparse.Namespace,
    ) -> argparse.Namespace:
        args.bindAddress = args.bindAddress or "[::]:50051"
        args.gpu = bool(args.gpu or False)
        args.servers = int(args.servers or 1)
        args.listeners = int(args.listeners or 1)
        args.workers = int(args.workers or 2)
        args.decoding_type = args.decoding_type or "GLOBAL"
        args.lm_algorithm = args.lm_algorithm or "viterbi"
        args.lm_weight = float(args.lm_weight or 0.2)
        args.word_score = float(args.word_score or -1)
        args.sil_score = float(args.sil_score or 0)
        args.overlap = int(args.overlap or 0)
        args.subwords = bool(args.subwords or False)
        args.local_formatting = bool(args.local_formatting or False)
        return args

    def checkArgsRequired(args: argparse.Namespace) -> argparse.Namespace:
        if not args.model:
            (
                cpu_dict_path,
                cpu_model_path,
                gpu_dict_path,
                gpu_model_path,
                standard_dict_path,
                standard_model_path,
            ) = Asr4ArgParser.setStandardModelPaths(args)
            Asr4ArgParser.constructModelPaths(
                args,
                cpu_dict_path,
                cpu_model_path,
                gpu_dict_path,
                gpu_model_path,
                standard_dict_path,
                standard_model_path,
            )

            if not (args.model or args.vocabulary):
                raise ValueError(
                    "No model/dict was specified and it couldn't be found on the standard paths/naming"
                )

        if args.lm_algorithm == "kenlm" and not (args.lm_model or args.lexicon):
            (
                lm_lexicon_path,
                lm_model_path,
                lm_version_lexicon_path,
                lm_version_model_path,
            ) = Asr4ArgParser.setStandardLMPaths(args)
            Asr4ArgParser.constructLMPaths(
                args,
                lm_lexicon_path,
                lm_model_path,
                lm_version_lexicon_path,
                lm_version_model_path,
            )

        if args.local_formatting and (
            not args.formatter or args.lm_algorithm != "kenlm"
        ):
            raise ValueError(
                "Local formatting was specified but no formatter model was given or lm algorithm is not kenlm"
            )

        return args

    def setStandardLMPaths(args):
        lm_model_path = f"asr4-{args.language.lower()}-lm.bin"
        lm_lexicon_path = f"asr4-{args.language.lower()}-lm.lexicon.txt"
        lm_version_model_path = f"asr4-{args.language.lower()}-lm-{args.lm_version}.bin"
        lm_version_lexicon_path = (
            f"asr4-{args.language.lower()}-lm-{args.lm_version}.lexicon.txt"
        )
        return (
            lm_lexicon_path,
            lm_model_path,
            lm_version_lexicon_path,
            lm_version_model_path,
        )

    def constructLMPaths(
        args,
        lm_lexicon_path,
        lm_model_path,
        lm_version_lexicon_path,
        lm_version_model_path,
    ):
        if os.path.exists(lm_model_path) and os.path.exists(lm_lexicon_path):
            args.lm_model = lm_model_path
            args.lexicon = lm_lexicon_path
        elif os.path.exists(lm_version_model_path) and os.path.exists(
            lm_version_lexicon_path
        ):
            args.lm_model = lm_version_model_path
            args.lexicon = lm_version_lexicon_path
        if args.lm_algorithm == "kenlm" and not (args.lm_model or args.lexicon):
            raise ValueError(
                "KenLM Language was specified but no Lexicon/LM could be found on the standards path naming"
            )

    def setStandardModelPaths(args):
        standard_model_path = f"asr4-{args.language.lower()}.onnx"
        standard_dict_path = "dict.ltr.txt"
        gpu_model_path = f"asr4-{args.language.lower()}-{args.gpu_version}.onnx"
        gpu_dict_path = f"asr4-{args.language.lower()}-{args.gpu_version}.dict.ltr.txt"
        cpu_model_path = f"asr4-{args.language.lower()}-{args.cpu_version}.onnx"
        cpu_dict_path = f"asr4-{args.language.lower()}-{args.cpu_version}.dict.ltr.txt"
        return (
            cpu_dict_path,
            cpu_model_path,
            gpu_dict_path,
            gpu_model_path,
            standard_dict_path,
            standard_model_path,
        )

    def constructModelPaths(
        args,
        cpu_dict_path,
        cpu_model_path,
        gpu_dict_path,
        gpu_model_path,
        standard_dict_path,
        standard_model_path,
    ):
        if os.path.exists(standard_model_path) and os.path.exists(standard_dict_path):
            args.model = standard_model_path
            args.vocabulary = standard_dict_path
        elif (
            args.gpu
            and os.path.exists(gpu_model_path)
            and os.path.exists(gpu_dict_path)
        ):
            args.model = gpu_model_path
            args.vocabulary = gpu_dict_path
        elif os.path.exists(cpu_model_path) and os.path.exists(cpu_dict_path):
            args.model = cpu_model_path
            args.vocabulary = cpu_dict_path


if __name__ == "__main__":
    main()
