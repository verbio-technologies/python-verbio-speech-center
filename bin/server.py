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
    args = fixNumberOfJobs(_parseArguments(sys.argv[1:]))
    args = SystemVarsOverride(args)
    args = TomlConfigurationOverride(args)
    args = checkArgsRequired(args)
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


def _parseArguments(args: list) -> argparse.Namespace:
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
    return parser.parse_args(args)


def setDefaultBindAddress(args, config):
    if not args.bindAddress:
        if os.environ.get("ASR4_HOST") and os.environ.get("ASR4_PORT"):
            args.bindAddress = (
                f"{os.environ.get('ASR4_HOST')}:{os.environ.get('ASR4_PORT')}"
            )
        else:
            config["global"].setdefault("host", "[::]")
            config["global"].setdefault("port", 50051)
            args.bindAddress = f"{config['global']['host']}:{config['global']['port']}"
            del config["global"]["host"]
            del config["global"]["port"]


def TomlConfigurationOverride(args: argparse.Namespace) -> argparse.Namespace:
    config_file = args.config or "asr4_config.toml"
    args.language = args.language or Language.EN_US.value

    if os.path.exists(config_file):
        config = toml.load(config_file)
        config.setdefault("global", {})
        setDefaultBindAddress(args, config)

        for k, v in config["global"].items():
            setattr(args, k, getattr(args, k, None) or v)

        if args.language.lower() in config:
            for k, v in config[args.language.lower()].items():
                setattr(args, k, getattr(args, k, None) or v)
    else:
        args.bindAddress = args.bindAddress or "[::]:50051"

    return args


def SystemVarsOverride(args: argparse.Namespace) -> argparse.Namespace:
    args.verbose = args.verbose or os.environ.get(
        "LOG_LEVEL", LoggerService.getDefaultLogLevel()
    )
    args.gpu = bool(args.gpu or os.environ.get("ASR4_GPU", False))
    args.servers = int(args.servers or os.environ.get("ASR4_SERVERS", "1"))
    args.listeners = int(args.listeners or os.environ.get("ASR4_LISTENERS", "1"))
    args.workers = int(args.workers or os.environ.get("ASR4_WORKERS", "2"))
    args.decoding_type = args.decoding_type or os.environ.get(
        "ASR4_DECODING_TYPE", "GLOBAL"
    )
    args.lm_algorithm = args.lm_algorithm or os.environ.get(
        "ASR4_LM_ALGORITHM", "viterbi"
    )
    args.lm_weight = float(args.lm_weight or os.environ.get("ASR4_LM_WEIGHT", "0.2"))
    args.word_score = float(args.word_score or os.environ.get("ASR4_WORD_SCORE", "-1"))
    args.sil_score = float(args.sil_score or os.environ.get("ASR4_SIL_SCORE", "0"))

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
        ) = setStandardModelPaths(args)

        constructModelPaths(
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
        ) = setStandardLMPaths(args)
        constructLMPaths(
            args,
            lm_lexicon_path,
            lm_model_path,
            lm_version_lexicon_path,
            lm_version_model_path,
        )

    return args


def constructLMPaths(
    args, lm_lexicon_path, lm_model_path, lm_version_lexicon_path, lm_version_model_path
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
    elif args.gpu and os.path.exists(gpu_model_path) and os.path.exists(gpu_dict_path):
        args.model = gpu_model_path
        args.vocabulary = gpu_dict_path
    elif os.path.exists(cpu_model_path) and os.path.exists(cpu_dict_path):
        args.model = cpu_model_path
        args.vocabulary = cpu_dict_path


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


def fixNumberOfJobs(args):
    if args.jobs is not None:
        args.servers = 1
        args.workers = 0
        args.listeners = args.jobs
    return args


if __name__ == "__main__":
    main()
