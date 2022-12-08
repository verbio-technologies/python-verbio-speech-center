import argparse

import grpc
import asyncio
from concurrent import futures
import multiprocessing
from grpc_health.v1 import health
from grpc_health.v1.health_pb2 import HealthCheckResponse
from grpc_health.v1.health_pb2_grpc import add_HealthServicer_to_server

from .types import SERVICES_NAMES
from .types import add_RecognizerServicer_to_server

from .formatter import FormatterFactory

from .service import RecognitionServiceConfiguration
from .service import RecognizerService

from .loggerService import LoggerQueue


class ServerConfiguration:
    def __init__(self, arguments: argparse.Namespace):
        self.bindAddress = arguments.bindAddress
        self.numberOfServers = arguments.servers
        self.numberOfListeners = arguments.listeners
        self.serviceConfiguration = RecognitionServiceConfiguration(arguments)

    def getServiceConfiguration(self):
        return self.serviceConfiguration


class Server:
    def __init__(self, configuration: ServerConfiguration, loggerService):
        self.loggerService = loggerService
        self._logger = loggerService.getLogger()
        self._server = None
        self._configuration = configuration

    def spawn(self):
        self._logger.info("Spawning server process.")
        self._server = multiprocessing.Process(
            target=Server._asyncRunServer,
            args=(self.loggerService.getQueue(), self.createGRpcServer(self._configuration))
        )
        self._server.start()
        self._logger.info("Server started")

    def join(self):
        if self._server is not None:
            self._server.join()

    @staticmethod
    def _asyncRunServer(
            logsQueue: LoggerQueue, gRpcServer: grpc.aio.server
    ) -> None:
        logsQueue.configureGlobalLogger()
        logger = logsQueue.getLogger()
        logger.info("Running asyncio server")
        asyncio.run(Server._runGRpcServer(logsQueue, gRpcServer))

    @staticmethod
    async def _runGRpcServer(
        logsQueue: LoggerQueue, gRpcServer: grpc.aio.server
    ) -> None:
        logsQueue.configureGlobalLogger()
        logger = logsQueue.getLogger()
        logger.info("Sever started")
        await gRpcServer.start()
        await gRpcServer.wait_for_termination()

    def createGRpcServer(self, configuration) -> grpc.aio.server:
        grpcServer = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=configuration.numberOfListeners),
            options=(("grpc.so_reuseport", 1),),
        )
        Server._addRecognizerService(grpcServer, configuration.getServiceConfiguration())
        Server._addHealthCheckService(grpcServer, configuration.numberOfListeners)
        grpcServer.add_insecure_port(configuration.bindAddress)
        self._logger.info(
            "Creating server with %d listeners, listening on %s", configuration.numberOfListeners, configuration.bindAddress
        )
        return grpcServer

    @staticmethod
    def _addRecognizerService(
        server: grpc.aio.Server, configuration: RecognitionServiceConfiguration
    ) -> None:
        add_RecognizerServicer_to_server(
            RecognizerService(
                configuration,
                FormatterFactory.createFormatter(
                    configuration.formatterModelPath, configuration.language
                )
                if configuration.formatterModelPath
                else None,
            ),
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
