#!/usr/bin/env python3
# Used to make sure python find proto files
import sys
sys.path.insert(1, '../proto/generated')

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Iterator
from typing import Iterable
from pprint import pprint
import argparse
import logging
import wave
import grpc

import recognition_pb2_grpc
import recognition_streaming_request_pb2
import recognition_streaming_response_pb2

class Options:
    def __init__(self):
        self.token_file = None
        self.host = "eu.speechcenter.verbio.com"
        self.audio_file = None
        self.topic = None
        self.language = 'en-US'
        self.sample_rate = 16000
        self.secure_channel = True

    def check(self):
        if self.topic is None:
            raise Exception("You must provide a least a topic")


def parse_command_line() -> Options:
    options = Options()
    parser = argparse.ArgumentParser(description='Perform speech recognition on an audio file')
    parser.add_argument('--audio-file', '-a', help='Path to a .wav audio in 8kHz and PCM16 encoding', required=True)
    argGroup = parser.add_mutually_exclusive_group(required=True)
    argGroup.add_argument('--topic', '-T', choices=['GENERIC', 'TELCO', 'BANKING', 'INSURANCE'], help='A valid topic')
    parser.add_argument('--language', '-l', choices=['en-US', 'pt-BR', 'es-ES'], help='A Language ID (default: ' + options.language + ')', default=options.language)
    parser.add_argument('--token', '-t', help='File with the authentication token', required=True)
    parser.add_argument('--sample-rate', '-s', help='Sample rate for audio: Example 8000 or 16000', required=True)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach (default: ' + options.host + ')',
                        default=options.host)
    parser.add_argument('--not-secure', '-S', help='Do not use a secure channel. Used for internal testing.', required=False, default=True, dest='secure', action='store_false')
    
    args = parser.parse_args()
    options.token_file = args.token
    options.host = args.host
    options.audio_file = args.audio_file
    options.topic = args.topic
    options.language = args.language
    options.sample_rate = int(args.sample_rate)
    options.secure_channel = args.secure
    '''
    if args.not_secure is not None:
        options.secure_channel = False
    else:
        options.secure_channel = True
    '''
    return options

class Credentials:
    def __init__(self, token):
        # Set JWT token for service access.
        self.call_credentials = grpc.access_token_call_credentials(token)
        # Set CA Certificate for SSL channel encryption.
        self.ssl_credentials = grpc.ssl_channel_credentials()

    def get_channel_credentials(self):
        return grpc.composite_channel_credentials(self.ssl_credentials, self.call_credentials)


class Resources:
    def __init__(self, options: Options):
        self.audio = self.__read_audio_file(options.audio_file)

    @staticmethod
    def __read_audio_file(audio_file: str) -> bytes:
        with open(audio_file, "rb") as wav_file:
            wav_data = wave.open(wav_file)
            audio_data = wav_data.readframes(wav_data.getnframes())
            return audio_data

class SpeechCenterStreamingASRClient:
    def __init__(self, executor: ThreadPoolExecutor, channel: grpc.Channel, options: Options):
        self._executor = executor
        self._channel = channel
        self._stub = recognition_pb2_grpc.RecognizerStub(self._channel)
        self._resources = Resources(options)
        self._host = options.host
        self._topic = options.topic
        self._sample_rate = options.sample_rate
        self._language = options.language
        self._peer_responded = threading.Event()
        self._credentials = Credentials(self.read_token(toke_file=options.token_file))
        self.token = self.read_token(toke_file=options.token_file)
        self._secure_channel = options.secure_channel

    def _response_watcher(
            self,
            response_iterator: Iterator[recognition_streaming_response_pb2.RecognitionStreamingResponse]) -> None:
        try:
            logging.info("Running response watcher")
            for response in response_iterator:
                logging.info("New incoming response %s", pprint(response))
            
        except Exception as e:
            logging.error("Error running response watcher: %s", str(e))
            self._peer_responded.set()
            raise

    @staticmethod
    def read_token(toke_file: str) -> str:
        with open(toke_file) as token_hdl:
            return ''.join(token_hdl.read().splitlines())
    
    def call(self) -> None:
        metadata = [('authorization', "Bearer " + self.token)]
        if self._secure_channel:
            response_iterator = self._stub.StreamingRecognize(self.__generate_inferences(topic=self._topic, wav_audio=self._resources.audio, language=self._language, sample_rate=self._sample_rate))
        else:
            response_iterator = self._stub.StreamingRecognize(self.__generate_inferences(topic=self._topic, wav_audio=self._resources.audio, language=self._language, sample_rate=self._sample_rate), metadata=metadata)
        
        self._consumer_future = self._executor.submit(self._response_watcher, response_iterator)
    
    def wait_server(self) -> bool:
        logging.info("Waiting for server to connect...")
        self._peer_responded.wait(timeout=None)
        if self._consumer_future.done():
            # If the future raises, forwards the exception here
            self._consumer_future.result()
        True
    
    @staticmethod
    def __generate_inferences(
        wav_audio: bytes,
        topic: str = "",
        language: str = "",
        sample_rate: int = 16000
    ) -> Iterable[recognition_streaming_request_pb2.RecognitionStreamingRequest]:
        """
        Inferences always start with a topic and a language, then audio is passed in a second message
        """
        if len(topic):
            var_resource = recognition_streaming_request_pb2.RecognitionResource(topic=0)
        else:
            raise Exception("Topic must be declared in order to perform the recognition")

        messages = [
            ("config", 
                recognition_streaming_request_pb2.RecognitionStreamingRequest(
                    config=recognition_streaming_request_pb2.RecognitionConfig(
                        parameters=recognition_streaming_request_pb2.RecognitionParameters(
                            language=language,
                            pcm=recognition_streaming_request_pb2.PCM(sample_rate_hz=sample_rate)
                        ), 
                        resource=var_resource))),
            ("audio", recognition_streaming_request_pb2.RecognitionStreamingRequest(audio=wav_audio)),
        ]
        for message_type, message in messages:
            logging.info("Sending streaming message " + message_type)
            yield message

    
def process_recognition(executor: ThreadPoolExecutor, channel: grpc.Channel, options: Options) -> None:
    client = SpeechCenterStreamingASRClient(executor, channel, options)
    client.call()
    logging.info("Press CTRL+C to exit")
    if client.wait_server():
        logging.info("Recognition started")
        client.audio_session()
        logging.info("Recognition finished")
    else:
        logging.error("Recognition failed: server didn't answer")


def run(command_line_options):
    executor = ThreadPoolExecutor()
    logging.info("Connecting to %s", command_line_options.host)

    if command_line_options.secure_channel:
        token = SpeechCenterStreamingASRClient.read_token(toke_file=command_line_options.token_file)
        credentials = Credentials(token)
        
        with grpc.secure_channel(command_line_options.host, credentials=credentials.get_channel_credentials()) as channel:
            logging.info("Running executor...")
            future = executor.submit(process_recognition, executor, channel, command_line_options)
            future.result()
            logging.info("New result arrived")
    else:
        with grpc.insecure_channel(command_line_options.host) as channel:
            logging.info("Running executor...")
            future = executor.submit(process_recognition, executor, channel, command_line_options)
            future.result()
            logging.info("New result arrived")
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Running speechcenter streaming channel...")
    command_line_options = parse_command_line()
    command_line_options.check()
    run(command_line_options)
    
