#!/usr/bin/env python3
# Used to make sure python find proto files
import sys
import time

sys.path.insert(1, '../proto/generated')

from concurrent.futures import ThreadPoolExecutor
import threading
from threading import Timer
from typing import Iterator, Iterable
import argparse
import logging
import grpc
from speechcenterauth import SpeechCenterCredentials
import recognition_pb2_grpc, recognition_streaming_request_pb2, recognition_streaming_response_pb2

from google.protobuf.json_format import MessageToJson
import io


class Options:
    def __init__(self):
        self.token_file = None
        self.host = ""
        self.input_file = None
        self.topic = None
        self.language = 'en-US'
        self.secure_channel = True
        self.formatting = False
        self.inactivity_timeout = False
        self.asr_version = None
        self.label = None
        self.client_id = None
        self.client_secret = None
        self.chunks = None
        self.sample_rate = None

    def check(self):
        if self.topic is None:
            raise Exception("You must provide a least a topic")


def parse_credential_args(args, options):
    if args.client_id and not args.client_secret:
        raise argparse.ArgumentError(None, "If --client-id is specified, then --client-secret must also be specified.")
    elif args.client_secret and not args.client_id:
        raise argparse.ArgumentError(None, "If --client-secret is specified, then --client-id must also be specified.")
    options.client_id = args.client_id or None
    options.client_secret = args.client_secret or None


def parse_command_line() -> Options:
    options = Options()
    parser = argparse.ArgumentParser(description='Perform speech recognition on an audio file')
    topicGroup = parser.add_mutually_exclusive_group(required=True)
    topicGroup.add_argument('--topic', '-T', choices=['GENERIC', 'TELCO', 'BANKING', 'INSURANCE'], help='A valid topic')
    parser.add_argument('--language', '-l',
                        choices=['en', 'en-US', 'en-GB', 'pt-BR', 'es', 'es-419', 'tr', 'ja', 'fr', 'fr-CA', 'de',
                                 'it'], help='A Language ID (default: ' + options.language + ')',
                        default=options.language)
    parser.add_argument('--token', '-t', help='File with the authentication token', required=True)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach (default: ' + options.host + ')',
                        required=True)
    parser.add_argument('--not-secure', '-S', help='Do not use a secure channel. Used for internal testing.',
                        required=False, default=True, dest='secure', action='store_false')
    parser.add_argument('--diarization', '-d', help='', required=False, default=False, action='store_false')
    parser.add_argument('--formatting', '-f', help='', required=False, default=False, action='store_false')
    parser.add_argument('--inactivity-timeout', '-i', help='Time for stream inactivity after the first valid response',
                        required=False, default=20.0)
    parser.add_argument('--asr-version', choices=['V1', 'V2'], help='Selectable asr version', required=True)
    parser.add_argument('--label', help='Label for the request', required=False, default="")
    parser.add_argument('--input-file', help='Input file descriptor', required=True)
    parser.add_argument('--chunks', help='Chunks of input', required=False, default=400)
    parser.add_argument('--sample-rate', help='audio sample rate', required=True)

    credentialGroup = parser.add_argument_group('credentials',
                                                '[OPTIONAL] Client authentication credentials used to refresh the token. You can find your credentials on the dashboard at https://dashboard.speechcenter.verbio.com/access-token')
    credentialGroup.add_argument('--client-id',
                                 help='Client id for authentication. MUST be written as --client-id=CLIENT_ID')
    credentialGroup.add_argument('--client-secret',
                                 help='Client secret for authentication. MUST be written as --client-secret=CLIENT_SECRET')

    args = parser.parse_args()
    parse_credential_args(args, options)

    options.token_file = args.token
    options.host = args.host
    options.input_file = args.input_file
    options.topic = args.topic
    options.language = args.language
    options.secure_channel = args.secure
    options.formatting = args.formatting
    options.diarization = args.diarization
    options.inactivity_timeout = float(args.inactivity_timeout)
    options.asr_version = args.asr_version
    options.label = args.label
    options.chunks = args.chunks
    options.sample_rate = int(args.sample_rate)

    return options


class GrpcChannelCredentials:
    def __init__(self, token):
        # Set JWT token for service access.
        self.call_credentials = grpc.access_token_call_credentials(token)
        # Set CA Certificate for SSL channel encryption.
        self.ssl_credentials = grpc.ssl_channel_credentials()

    def get_channel_credentials(self):
        return grpc.composite_channel_credentials(self.ssl_credentials, self.call_credentials)


class SpeechCenterStreamingASRClient:
    def __init__(self, executor: ThreadPoolExecutor, channel: grpc.Channel, options: Options):
        self._executor = executor
        self._channel = channel
        self._stub = recognition_pb2_grpc.RecognizerStub(self._channel)
        self._host = options.host
        self._topic = options.topic
        self._language = options.language
        self._peer_responded = threading.Event()
        self.token = retrieve_token(options)
        self._secure_channel = options.secure_channel
        self._inactivity_timer = None
        self._inactivity_timer_timeout = options.inactivity_timeout
        self._asr_version = options.asr_version
        self._formatting = options.formatting
        self._diarization = options.diarization
        self._label = options.label
        self.__file = options.input_file
        self.__chunks = options.chunks
        self.__sample_rate = options.sample_rate

    def _close_stream_by_inactivity(self):
        logging.info("Stream inactivity detected, closing stream...")
        self._peer_responded.set()

    def _start_inactivity_timer(self, inactivity_timeout: float):
        self._inactivity_timer = Timer(inactivity_timeout, self._close_stream_by_inactivity)
        self._inactivity_timer.start()

    def _print_result(self, response):
        duration = response.result.duration
        for alternative in response.result.alternatives:
            if alternative.transcript:
                print('\t"transcript": "%s",\n\t"confidence": "%f",\n\t"duration": "%f"' % (
                    alternative.transcript, alternative.confidence, duration))

    def _response_watcher(
            self,
            response_iterator: Iterator[recognition_streaming_response_pb2.RecognitionStreamingResponse]) -> None:
        try:
            logging.info("Running response watcher")
            for response in response_iterator:
                json = MessageToJson(response)
                logging.info("New incoming response: '%s ...'", json[0:50].replace('\n', ''))
                self._print_result(response)

                if response.result and response.result.is_final:
                    if self._inactivity_timer:
                        self._inactivity_timer.cancel()
                    self._start_inactivity_timer(self._inactivity_timer_timeout)

        except Exception as e:
            logging.error("Error running response watcher: %s", str(e))
            self._peer_responded.set()
            raise

    def call(self) -> None:
        metadata = [('authorization', "Bearer " + self.token)]
        if self._secure_channel:
            request_iterator = self._stub.StreamingRecognize(
                self.__generate_inferences(topic=self._topic, asr_version=self._asr_version, file=self.__file,
                                           chunks=self.__chunks, language=self._language,
                                           sample_rate=self.__sample_rate, formatting=self._formatting,
                                           diarization=self._diarization, label=self._label))
        else:
            request_iterator = self._stub.StreamingRecognize(
                self.__generate_inferences(topic=self._topic, asr_version=self._asr_version, file=self.__file,
                                           chunks=self.__chunks, language=self._language,
                                           sample_rate=self.__sample_rate, formatting=self._formatting,
                                           diarization=self._diarization, label=self._label), metadata=metadata)

        self._consumer_future = self._executor.submit(self._response_watcher, request_iterator)

    def wait_server(self) -> bool:
        logging.info("Waiting for server to respond...")
        self._peer_responded.wait(timeout=None)
        if self._consumer_future.done():
            # If the future raises, forwards the exception here
            self._consumer_future.result()

        return True

    @staticmethod
    def divide_audio(file, chunks, sample_rate):
        file = io.open(file, 'rb', -1)

        time.sleep(chunks / sample_rate)
        audio = file.read(chunks)
        while 1:
            if not audio:
                break
            yield audio
            time.sleep(chunks / sample_rate)
            audio = file.read(chunks)

        file.close()

    @staticmethod
    def __generate_inferences(
            file: int,
            chunks: int,
            asr_version: str,
            topic: str = "",
            language: str = "",
            sample_rate: int = 16000,
            diarization=False,
            formatting=False,
            label: str = "",
    ) -> Iterable[recognition_streaming_request_pb2.RecognitionStreamingRequest]:

        if len(topic):
            var_resource = recognition_streaming_request_pb2.RecognitionResource(topic=topic)
        else:
            raise Exception("Topic must be declared in order to perform the recognition")

        if len(asr_version):
            asr_versions = {"V1": 0, "V2": 1}
            selected_asr_version = asr_versions[asr_version]
        else:
            raise Exception("ASR version must be declared in order to perform the recognition")

        yield recognition_streaming_request_pb2.RecognitionStreamingRequest(
            config=recognition_streaming_request_pb2.RecognitionConfig(
                parameters=recognition_streaming_request_pb2.RecognitionParameters(
                    language=language,
                    pcm=recognition_streaming_request_pb2.PCM(sample_rate_hz=sample_rate),
                    enable_formatting=formatting,
                    enable_diarization=diarization
                ),
                resource=var_resource,
                label=[label],
                version=selected_asr_version
            )
        )

        for chunk in SpeechCenterStreamingASRClient.divide_audio(file, chunks, sample_rate):
            logging.debug("Appending chunk as message: " + repr(chunk)[0:20] + "...")
            yield recognition_streaming_request_pb2.RecognitionStreamingRequest(audio=chunk)

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


def retrieve_token(options):
    if options.client_id:
        return SpeechCenterCredentials.get_refreshed_token(options.client_id, options.client_secret, options.token_file)
    else:
        return SpeechCenterCredentials.read_token(token_file=options.token_file)


def run(command_line_options):
    executor = ThreadPoolExecutor()
    logging.info("Connecting to %s", command_line_options.host)

    if command_line_options.secure_channel:
        token = retrieve_token(command_line_options)
        credentials = GrpcChannelCredentials(token)
        with grpc.secure_channel(command_line_options.host,
                                 credentials=credentials.get_channel_credentials()) as channel:
            run_executor(command_line_options, executor, channel)

    else:
        with grpc.insecure_channel(command_line_options.host) as channel:
            run_executor(command_line_options, executor, channel)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]:%(message)s')
    logging.info("Running speechcenter streaming channel...")
    command_line_options = parse_command_line()
    command_line_options.check()
    run(command_line_options)
