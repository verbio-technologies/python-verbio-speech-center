#!/usr/bin/env python3
import sys
import logging
from typing import Optional
from rich.logging import RichHandler
sys.path.insert(1, '../proto/generated')

import grpc
import recognition_pb2_grpc
from concurrent.futures import ThreadPoolExecutor

from helpers.csr_gui import CsrGUI
from helpers.csr_client import CSRClient
from helpers.audio_importer import AudioImporter
from helpers.grpc_connection import GrpcConnection
from helpers.common import retrieve_token, parse_csr_commandline, RecognizerOptions


def process_recognition(executor: ThreadPoolExecutor, channel: grpc.Channel, options: RecognizerOptions, access_token: str, gui: Optional[CsrGUI]):
    audio_resource = AudioImporter(options.audio_file, options.convert_audio)
    stub = recognition_pb2_grpc.RecognizerStub(channel)
    client = CSRClient(executor, stub, options, audio_resource, access_token, gui)
    client.send_audio()
    client.wait_for_response()
    logging.info("Recognition finished")


def init_gui() -> CsrGUI:
    gui = CsrGUI()
    logging_handler = RichHandler(
        console=gui._logging_console,
        show_time=False,
        show_level=False,
        show_path=False
    )
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s][%(levelname)s]:%(message)s',
        handlers=[logging_handler],
        force=True
    )
    gui.start()
    return gui


def run(options: RecognizerOptions):
    gui = init_gui() if options.gui else None
    access_token = retrieve_token(command_line_options)
    grpc_connection = GrpcConnection(options.secure_channel, options.client_id, options.client_secret, access_token)

    with grpc_connection.open(options.host) as grpc_channel:
        executor = ThreadPoolExecutor()
        future = executor.submit(process_recognition, executor, grpc_channel, options, access_token, gui)
        future.result()

    if gui:
        gui.stop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]:%(message)s')
    command_line_options = parse_csr_commandline()
    command_line_options.check()
    run(command_line_options)
