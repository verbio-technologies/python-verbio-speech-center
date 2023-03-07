#!/usr/bin/env python3
# Used to make sure python find proto files
import sys
sys.path.insert(1, '../proto/generated')

from concurrent.futures import ThreadPoolExecutor
import threading
from threading import Timer
from typing import Iterator, Iterable
import argparse
import logging
import wave
import grpc

import recognition_pb2_grpc
import recognition_streaming_request_pb2
import recognition_streaming_response_pb2

from google.protobuf.json_format import MessageToJson


class Options:
    def __init__(self):
        self.token_file = None
        self.host = ""
        self.audio_file = None
        self.topic = None
        self.language = 'en-US'
        self.secure_channel = True
        self.formatting = False
        self.inactivity_timeout = False
        self.asr_version = None

    def check(self):
        if self.topic is None:
            raise Exception("You must provide a least a topic")


def parse_command_line() -> Options:
    options = Options()
    parser = argparse.ArgumentParser(description='Perform speech recognition on an audio file')
    parser.add_argument('--audio-file', '-a', help='Path to a .wav audio in 8kHz and PCM16 encoding', required=True)
    argGroup = parser.add_mutually_exclusive_group(required=True)
    argGroup.add_argument('--topic', '-T', choices=['GENERIC', 'TELCO', 'BANKING', 'INSURANCE'], help='A valid topic')
    parser.add_argument('--language', '-l', choices=['en-US', 'en-GB', 'pt-BR', 'es', 'es-419', 'tr', 'ja', 'fr', 'fr-CA', 'de', 'it'], help='A Language ID (default: ' + options.language + ')', default=options.language)
    parser.add_argument('--token', '-t', help='File with the authentication token', required=True)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach (default: ' + options.host + ')', required=True)
    parser.add_argument('--not-secure', '-S', help='Do not use a secure channel. Used for internal testing.', required=False, default=True, dest='secure', action='store_false')
    parser.add_argument('--diarization', '-d', help='', required=False, default=False, action='store_false')
    parser.add_argument('--formatting', '-f', help='', required=False, default=False, action='store_false')
    parser.add_argument('--inactivity-timeout', '-i', help='Time for stream inactivity after the first valid response', required=False, default=5.0)
    parser.add_argument('--asr-version', choices=['V1', 'V2'], help='Selectable asr version', required=True)
    
    args = parser.parse_args()
    options.token_file = args.token
    options.host = args.host
    options.audio_file = args.audio_file
    options.topic = args.topic
    options.language = args.language
    options.secure_channel = args.secure
    options.formatting = args.formatting
    options.diarization = args.diarization
    options.inactivity_timeout = float(args.inactivity_timeout)
    options.asr_version = args.asr_version
    
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
        with open(options.audio_file, "rb") as wav_file:
            wav_data = wave.open(wav_file)
            self.sample_rate = wav_data.getframerate()
            self.audio = wav_data.readframes(wav_data.getnframes())
            wav_data.close()


class SpeechCenterStreamingASRClient:
    def __init__(self, executor: ThreadPoolExecutor, channel: grpc.Channel, options: Options):
        self._executor = executor
        self._channel = channel
        self._stub = recognition_pb2_grpc.RecognizerStub(self._channel)
        self._resources = Resources(options)
        self._host = options.host
        self._topic = options.topic
        self._language = options.language
        self._peer_responded = threading.Event()
        self._credentials = Credentials(self.read_token(toke_file=options.token_file))
        self.token = self.read_token(toke_file=options.token_file)
        self._secure_channel = options.secure_channel
        self._inactivity_timer = None
        self._inactivity_timer_timeout = options.inactivity_timeout
        self._asr_version = options.asr_version
        self._formatting = options.formatting
        self._diarization = options.diarization

    def _close_stream_by_inactivity(self):
        logging.info("Stream inactivity detected, closing stream...")
        self._peer_responded.set()

    def _start_inactivity_timer(self, inactivity_timeout : float):
        self._inactivity_timer = Timer(inactivity_timeout, self._close_stream_by_inactivity)
        self._inactivity_timer.start()

    def _response_watcher(
            self,
            response_iterator: Iterator[recognition_streaming_response_pb2.RecognitionStreamingResponse]) -> None:
        try:
            logging.info("Running response watcher")
            for response in response_iterator:
                json = MessageToJson(response)
                logging.info("New incoming response: '%s ...'", json[0:50].replace('\n', ''))
                print(MessageToJson(response))

                if response.result and response.result.is_final:
                    if self._inactivity_timer:
                        self._inactivity_timer.cancel()
                    self._start_inactivity_timer(self._inactivity_timer_timeout)

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
            response_iterator = self._stub.StreamingRecognize(self.__generate_inferences(topic=self._topic, asr_version=self._asr_version, wav_audio=self._resources.audio, language=self._language, sample_rate=self._resources.sample_rate, formatting=self._formatting, diarization=self._diarization))
        else:
            response_iterator = self._stub.StreamingRecognize(self.__generate_inferences(topic=self._topic, asr_version=self._asr_version, wav_audio=self._resources.audio, language=self._language, sample_rate=self._resources.sample_rate, formatting=self._formatting, diarization=self._diarization), metadata=metadata)
        
        self._consumer_future = self._executor.submit(self._response_watcher, response_iterator)
    
    def wait_server(self) -> bool:
        logging.info("Waiting for server to respond...")
        self._peer_responded.wait(timeout=None)
        if self._consumer_future.done():
            # If the future raises, forwards the exception here
            self._consumer_future.result()
        
        return True
    
    @staticmethod
    def __generate_inferences(
        wav_audio: bytes,
        asr_version: str,
        topic: str = "",
        language: str = "",
        sample_rate: int = 16000,
        diarization = False,
        formatting = False
    ) -> Iterable[recognition_streaming_request_pb2.RecognitionStreamingRequest]:
        
        if len(topic):
            var_resource = recognition_streaming_request_pb2.RecognitionResource(topic=topic)
        else:
            raise Exception("Topic must be declared in order to perform the recognition")
        
        if len(asr_version):
            asr_versions = {"V1":0, "V2":1}
            selected_asr_version = asr_versions[asr_version]
        else:
            raise Exception("ASR version must be declared in order to perform the recognition")

        messages = [
            ("config", 
                recognition_streaming_request_pb2.RecognitionStreamingRequest(
                    config=recognition_streaming_request_pb2.RecognitionConfig(
                        parameters=recognition_streaming_request_pb2.RecognitionParameters(
                            language=language,
                            pcm=recognition_streaming_request_pb2.PCM(sample_rate_hz=sample_rate),
                            enable_formatting = formatting,
                            enable_diarization = diarization
                        ), 
                        resource=var_resource,
                        version=selected_asr_version))),
            ("audio", recognition_streaming_request_pb2.RecognitionStreamingRequest(audio=wav_audio)),
        ]
        for message_type, message in messages:
            logging.info("Sending streaming message " + message_type)
            yield message
        logging.info("All audio messages sent")

    
def process_recognition(executor: ThreadPoolExecutor, channel: grpc.Channel, options: Options) -> None:
    client = SpeechCenterStreamingASRClient(executor, channel, options)
    client.call()
    if client.wait_server():
        logging.info("Recognition finished")
    else:
        logging.error("Recognition failed: server didn't answer")


def run_executor(command_line_options, executor, channel):
    logging.info("Running executor...")
    future = executor.submit(process_recognition, executor, channel, command_line_options)
    future.result()
 
def run(command_line_options):
    executor = ThreadPoolExecutor()
    logging.info("Connecting to %s", command_line_options.host)

    if command_line_options.secure_channel:
        token = SpeechCenterStreamingASRClient.read_token(toke_file=command_line_options.token_file)
        credentials = Credentials(token)
        with grpc.secure_channel(command_line_options.host, credentials=credentials.get_channel_credentials()) as channel:
            run_executor(command_line_options, executor, channel)
            
    else:
        with grpc.insecure_channel(command_line_options.host) as channel:
            run_executor(command_line_options, executor, channel)

   

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(levelname)s]:%(message)s')
    logging.info("Running speechcenter streaming channel...")
    command_line_options = parse_command_line()
    command_line_options.check()
    run(command_line_options)
    
