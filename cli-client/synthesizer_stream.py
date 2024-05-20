#!/usr/bin/env python3
import sys
sys.path.insert(1, '../proto/generated')
import grpc
import logging
from helpers.tts_client import TTSClient
import verbio_speech_center_synthesizer_pb2_grpc
from concurrent.futures import ThreadPoolExecutor
from helpers.grpc_connection import GrpcConnection
from helpers.common import SynthesizerOptions, parse_tts_command_line, retrieve_token


def process_synthesis(executor: ThreadPoolExecutor, channel: grpc.Channel, options: SynthesizerOptions, access_token: str):
    stub = verbio_speech_center_synthesizer_pb2_grpc.TextToSpeechStub(channel)
    client = TTSClient(executor, stub, options, access_token)
    if options.text:
        audio_samples = client.synthesize()
        client.save_audio_result(audio_samples)
    elif options.text_file:
        client.send_text()
        client.wait_for_response()
    logging.info("Synthesis finished")


def run(options: SynthesizerOptions):
    logging.info("Connecting to %s", command_line_options.host)
    access_token = retrieve_token(command_line_options)
    grpc_connection = GrpcConnection(options.secure_channel, options.client_id, options.client_secret, access_token)

    with grpc_connection.open(options.host) as grpc_channel:
        executor = ThreadPoolExecutor()
        future = executor.submit(process_synthesis, executor, grpc_channel, options, access_token)
        future.result()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]:%(message)s')
    logging.info("Running speechcenter TTS synthesis...")
    command_line_options = parse_tts_command_line()
    run(command_line_options)
