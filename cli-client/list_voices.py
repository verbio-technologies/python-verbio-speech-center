#!/usr/bin/env python3
import sys
sys.path.insert(1, './proto/generated')

import grpc
import logging
import argparse
import speechcenter.tts.text_to_speech_pb2_grpc as text_to_speech_pb2_grpc
from concurrent.futures import ThreadPoolExecutor
from helpers.grpc_connection import GrpcConnection
from helpers.tts_client import TTSClient
from helpers.common import SynthesizerOptions, parse_credential_args, retrieve_token


def parse_command_line():
    options = SynthesizerOptions()
    parser = argparse.ArgumentParser(description='List available voices for speech synthesis')
    parser.add_argument('--token', '-t', help='File with the authentication token', required=True)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach', required=True)
    parser.add_argument('--language', '-l', help='Filter voices by language (e.g. en-US, es-ES)', required=False, default=None)
    parser.add_argument('--not-secure', '-S', help='Do not use a secure channel. Used for internal testing.',
                        required=False, default=True, dest='secure', action='store_false')

    credential_group = parser.add_argument_group(
        'credentials',
        '''[OPTIONAL] Client authentication credentials used to refresh the token.
        You can find your credentials on the dashboard at https://dashboard.speechcenter.verbio.com/access-token''')
    credential_group.add_argument('--client-id', help='Client id for authentication. MUST be written as --client-id=CLIENT_ID')
    credential_group.add_argument('--client-secret', help='Client secret for authentication. MUST be written as --client-secret=CLIENT_SECRET')

    args = parser.parse_args()
    parse_credential_args(args, options)

    options.token_file = args.token
    options.host = args.host
    options.secure_channel = args.secure

    return options, args.language


def run(options: SynthesizerOptions, language: str):
    logging.info("Connecting to %s", options.host)
    access_token = retrieve_token(options)
    grpc_connection = GrpcConnection(options.secure_channel, options.client_id, options.client_secret, access_token)

    with grpc_connection.open(options.host) as grpc_channel:
        stub = text_to_speech_pb2_grpc.TextToSpeechStub(grpc_channel)
        client = TTSClient(ThreadPoolExecutor(), stub, options, access_token)
        voices = client.list_voices(language=language)

        print("Available voices:")
        for voice in voices:
            print(f"  {voice}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]:%(message)s')
    logging.info("Listing available TTS voices...")
    command_line_options, language = parse_command_line()
    run(command_line_options, language)
