import argparse
import asyncio
from concurrent import futures
import grpc
from loguru import logger
import multiprocessing
from grpc_health.v1 import health
from grpc_health.v1.health_pb2 import HealthCheckResponse
from grpc_health.v1.health_pb2_grpc import add_HealthServicer_to_server

from .types import SERVICES_NAMES
from .types import add_RecognizerServicer_to_server
from .service import RecognizerService


class ServerConfiguration:
    def __init__(self, arguments: argparse.Namespace):
        self.bindAddress = arguments.bindAddress
        self.numberOfServers = arguments.servers
        self.numberOfListeners = arguments.listeners
        self.serviceConfiguration = arguments.config

class Server:
    def __init__(self, configuration: ServerConfiguration):
        self._server = None
        self._configuration = configuration

    def spawn(self):
        logger.info("Spawning server process.")
        self._server = multiprocessing.Process(
            target=Server._asyncRunServer,
            args=(self._configuration,),
        )
        self._server.start()
        logger.info("Server started")

    def join(self):
        if self._server is not None:
            self._server.join()

    @staticmethod
    def _asyncRunServer(configuration: ServerConfiguration) -> None:
        logger.info(
            "Running gRPC server with %d listeners on %s"
            % (configuration.numberOfListeners, configuration.bindAddress)
        )
        asyncio.run(Server._runGRpcServer(configuration))

    @staticmethod
    async def _runGRpcServer(configuration: ServerConfiguration) -> None:
        gRpcServer = Server.createGRpcServer(configuration)
        await gRpcServer.start()
        logger.info("Server started")
        await gRpcServer.wait_for_termination()

    @staticmethod
    def createGRpcServer(configuration: ServerConfiguration) -> grpc.aio.server:
        grpcServer = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=configuration.numberOfListeners),
            options=(
                ("grpc.so_reuseport", 1),
                ("grpc.max_receive_message_length", -1),
            ),
        )
        Server._addRecognizerService(grpcServer, configuration.serviceConfiguration)
        Server._addHealthCheckService(grpcServer, configuration.numberOfListeners)
        grpcServer.add_insecure_port(configuration.bindAddress)
        return grpcServer

    @staticmethod
    def _addRecognizerService(server: grpc.aio.Server, configuration: str) -> None:
        add_RecognizerServicer_to_server(
            RecognizerService(configuration),
            server,
        )

    @staticmethod
    def _addHealthCheckService(
        server: grpc.aio.Server,
        jobs: int,
    ) -> None:
        healthServicer = health.HealthServicer(
            experimental_non_blocking=True,
            experimental_thread_pool=futures.ThreadPoolExecutor(max_workers=jobs),
        )
        Server._markAllServicesAsHealthy(healthServicer)
        add_HealthServicer_to_server(healthServicer, server)

    @staticmethod
    def _markAllServicesAsHealthy(healthServicer: health.HealthServicer) -> None:
        for service in SERVICES_NAMES + [health.SERVICE_NAME]:
            healthServicer.set(service, HealthCheckResponse.SERVING)
