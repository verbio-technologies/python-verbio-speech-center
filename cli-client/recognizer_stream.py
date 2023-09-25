import sys
sys.path.insert(1, '../proto/generated')
import grpc
import logging
import recognition_pb2_grpc
from helpers.csr_client import CSRClient
from helpers.audio_importer import AudioImporter
from concurrent.futures import ThreadPoolExecutor
from helpers.grpc_connection import GrpcConnection
from helpers.common import retrieve_token, parse_csr_commandline, RecognizerOptions


def process_recognition(executor: ThreadPoolExecutor, channel: grpc.Channel, options: RecognizerOptions, access_token: str):
    audio_resource = AudioImporter(options.audio_file)
    stub = recognition_pb2_grpc.RecognizerStub(channel)
    client = CSRClient(executor, stub, options, audio_resource, access_token)
    client.send_audio()
    client.wait_for_response()
    logging.info("Recognition finished")


def run(options):
    logging.info("Connecting to %s", command_line_options.host)
    access_token = retrieve_token(command_line_options)
    grpc_connection = GrpcConnection(options.secure_channel, options.client_id, options.client_secret, access_token)

    with grpc_connection.open(options.host) as grpc_channel:
        executor = ThreadPoolExecutor()
        future = executor.submit(process_recognition, executor, grpc_channel, options, access_token)
        future.result()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]:%(message)s')
    logging.info("Running speechcenter streaming...")
    command_line_options = parse_csr_commandline()
    command_line_options.check()
    run(command_line_options)
