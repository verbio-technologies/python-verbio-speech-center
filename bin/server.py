import argparse, os, sys, toml
from loguru import logger
import multiprocessing

from asr4_streaming.recognizer import Server, ServerConfiguration
from asr4_streaming.recognizer import Logger

from asr4_engine.data_classes import Language
from asr4.engines.wav2vec.v1.runtime.onnx import DecodingType


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = Asr4ArgParser(sys.argv[1:]).getArgs()
    _ = Logger(args.verbose)
    serve(ServerConfiguration(args))


def serve(
    configuration,
) -> None:
    servers = []
    for i in range(configuration.numberOfServers):
        logger.info("Starting server %s" % i)
        server = Server(configuration)
        server.spawn()
        servers.append(server)

    for server in servers:
        server.join()


class Asr4ArgParser:
    def __init__(self, argv):
        self.argv = argv

    def getArgs(self) -> argparse.Namespace:
        args = Asr4ArgParser.parseArguments(self.argv)
        args = Asr4ArgParser.replaceUndefinedWithEnvVariables(args)
        args = Asr4ArgParser.replaceUndefinedWithConfigFile(args)
        args = Asr4ArgParser.replaceUndefinedWithDefaultValues(args)
        args = Asr4ArgParser.fixNumberOfJobs(args)
        return args

    def parseArguments(args: list) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Python ASR4 Server")
        parser.add_argument(
            "-v",
            "--verbose",
            type=str,
            choices=Logger.getLevels(),
            help="Log levels. By default reads env variable LOG_LEVEL.",
        )
        parser.add_argument(
            "--host",
            dest="bindAddress",
            help="Hostname address to bind the server to.",
        )
        parser.add_argument(
            "-C", "--config", dest="config", help="Path to the asr4 config file"
        )
        return parser.parse_args(args)

    def replaceUndefinedWithEnvVariables(
        args: argparse.Namespace,
    ) -> argparse.Namespace:
        args.verbose = args.verbose or os.environ.get(
            "LOG_LEVEL", Logger.getDefaultLevel()
        )
        if os.environ.get("ASR4_HOST") and os.environ.get("ASR4_PORT"):
            args.bindAddress = (
                f"{os.environ.get('ASR4_HOST')}:{os.environ.get('ASR4_PORT')}"
            )

        return args

    def replaceUndefinedWithConfigFile(args: argparse.Namespace) -> argparse.Namespace:
        configFile = args.config or "asr4_config.toml"
        if os.path.exists(configFile):
            config = toml.load(configFile)
            config.setdefault("global", {})
            args = Asr4ArgParser.fillArgsFromTomlFile(args, config)
        return args

    def fillArgsFromTomlFile(args: argparse.Namespace, config):
        if config["global"].setdefault("host") and config["global"].setdefault("port"):
            args.bindAddress = f"{config['global']['host']}:{config['global']['port']}"
            del config["global"]["host"]
            del config["global"]["port"]
        for k, v in config["global"].items():
            setattr(args, k, getattr(args, k, None) or v)
        return args

    def replaceUndefinedWithDefaultValues(
        args: argparse.Namespace,
    ) -> argparse.Namespace:
        args.bindAddress = args.bindAddress or "[::]:50051"
        args.servers = int(args.servers) if "servers" in args else 1
        args.listeners = int(args.listeners) if "listeners" in args else 1
        args.workers = int(args.workers) if "workers" in args else 2
        return args

    def fixNumberOfJobs(args):
        if "jobs" in args:
            args.servers = 1
            args.workers = 0
            args.listeners = args.jobs
        return args


if __name__ == "__main__":
    main()
