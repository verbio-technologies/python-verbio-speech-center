import os

import argparse
import multiprocessing

from asr4.recognizer import Language
from asr4.recognizer import LoggerService
from asr4.recognizer import Server, ServerConfiguration
from asr4.recognizer import DecodingType


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = fixNumberOfJobs(_parseArguments())
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
        required=True,
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
        default=Language.EN_US.value,
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
        default=False,
        action="store_true",
        help="Whether to use GPU instead of CPU",
    )
    parser.add_argument(
        "--host",
        dest="bindAddress",
        default="[::]:50051",
        help="Hostname address to bind the server to.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        dest="jobs",
        default=None,
        help="Deprecated. Just for backcompatibility issues. Overrides -S, -L and -w and has the same effect as:  -S 1 -L {jobs} -w 0",
    )
    parser.add_argument(
        "-s",
        "--servers",
        type=int,
        dest="servers",
        default=1,
        help="The number of inference servers to be run. Each server will load a whole new inference system.",
    )
    parser.add_argument(
        "-L",
        "--listeners",
        type=int,
        dest="listeners",
        default=2,
        help="Number of gRPC listeners that a server will load. All listeners share the same inference server",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        dest="workers",
        default=2,
        help="The number of workers that a single listener can use to resolve a single request.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=str,
        choices=LoggerService.getLogLevelOptions(),
        default=os.environ.get("LOG_LEVEL", LoggerService.getDefaultLogLevel()),
        help="Log levels. By default reads env variable LOG_LEVEL.",
    )
    parser.add_argument(
        "-D",
        "--decoding-type",
        type=str,
        dest="decoding_type",
        choices=DecodingType._member_names_,
        default=os.environ.get("DECODING_TYPE", "GLOBAL"),
        help="Perform Decoding for each chunk (Local) or for all chunks (Global)",
    )
    parser.add_argument(
        "--lm-algorithm",
        type=str,
        dest="lm_algorithm",
        choices=["viterbi", "kenlm"],
        default="viterbi",
        help="Type of algorithm for language model decoding.",
    )
    parser.add_argument(
        "--lm-lexicon",
        type=str,
        dest="lexicon",
        required=False,
        help="Lexicon for language model.",
    )
    parser.add_argument(
        "--lm-model",
        type=str,
        dest="lm_model",
        required=False,
        help="Path to the language model file.",
    )
    return parser.parse_args()


def fixNumberOfJobs(args):
    if args.jobs is not None:
        args.servers = 1
        args.workers = 0
        args.listeners = args.jobs
    return args


if __name__ == "__main__":
    main()
