#!/usr/bin/env python3
import argparse
import grpc
import logging
from options import Options
from authentication import retrieveToken
from credentials import GrpcChannelCredentials
from concurrent.futures import ThreadPoolExecutor
from recognizer_stream import SpeechCenterStreamingASRClient

def setupArgParser():
    parser = argparse.ArgumentParser(description='Perform speech recognition on an audio file')
    parser.add_argument('--audio-file', '-a', help='Path to a .wav audio in 8kHz and PCM16 encoding', required=True)
    topicGroup = parser.add_mutually_exclusive_group(required=True)
    topicGroup.add_argument('--topic', '-T', choices=['GENERIC', 'TELCO', 'BANKING', 'INSURANCE'], help='A valid topic')
    parser.add_argument('--language', '-l', choices=['en', 'en-US', 'en-GB', 'pt-BR', 'es', 'es-419', 'tr', 'ja', 'fr', 'fr-CA', 'de', 'it'], help='A Language ID (default: ' + options.language + ')')
    parser.add_argument('--token', '-t', help='File with the authentication token', required=True)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach (default: ' + options.host + ')', required=True)
    parser.add_argument('--not-secure', '-S', help='Do not use a secure channel. Used for internal testing.', required=False, default=True, dest='secure', action='store_false')
    parser.add_argument('--diarization', '-d', help='', required=False, default=False, action='store_false')
    parser.add_argument('--formatting', '-f', help='', required=False, default=False, action='store_false')
    parser.add_argument('--inactivity-timeout', '-i', help='Time for stream inactivity after the first valid response', required=False, default=5.0)
    parser.add_argument('--asr-version', choices=['V1', 'V2'], help='Selectable asr version', required=True)
    parser.add_argument('--label', help='"Label for the request', required=False, default="")
    parser.add_argument('--logging-level', help='"Logging level of logs', required=False, default="INFO")
    
    credentialGroup = parser.add_argument_group('credentials', '[OPTIONAL] Client authentication credentials used to refresh the token. You can find your credentials on the dashboard at https://dashboard.speechcenter.verbio.com/access-token')
    credentialGroup.add_argument('--client-id', help='Client id for authentication. MUST be written as --client-id=CLIENT_ID')
    credentialGroup.add_argument('--client-secret', help='Client secret for authentication. MUST be written as --client-secret=CLIENT_SECRET')

    return parser

def parseArguments(parser: argparse.ArgumentParser) -> Options:
    args = parser.parse_args()
    options = Options(args)
    
    return options

def processAudioRecognition(channel: grpc.Channel, options: Options) -> None:
    client = SpeechCenterStreamingASRClient(channel, options)
    client.call()
    if client.wait_server():
        logging.info("Recognition finished")
    else:
        logging.error("Recognition failed: server didn't answer")

def runExecutor(command_line_options, executor, channel):
    logging.info("Running executor...")
    future = executor.submit(processAudioRecognition, executor, channel, command_line_options)
    future.result()

def run(command_line_options):
    executor = ThreadPoolExecutor()
    logging.info("Connecting to %s", command_line_options.host)

    if command_line_options.secure_channel:
        token = retrieveToken(command_line_options)
        credentials = GrpcChannelCredentials(token)
        with grpc.secure_channel(command_line_options.host, credentials=credentials.get_channel_credentials()) as channel:
            runExecutor(command_line_options, executor, channel)
            
    else:
        with grpc.insecure_channel(command_line_options.host) as channel:
            runExecutor(command_line_options, executor, channel)

if __name__ == '__main__':
    cli_options = parseArguments(setupArgParser())
    logging.basicConfig(level=cli_options.logging_level, format='[%(asctime)s][%(levelname)s]:%(message)s')
    logging.info("Running speechcenter streaming channel...")
    cli_options.check()
    run(cli_options)