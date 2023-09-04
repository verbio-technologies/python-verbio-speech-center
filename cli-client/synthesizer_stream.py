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
import math
import wave
import grpc
from speechcenterauth import SpeechCenterCredentials
import verbio_speech_center_synthesizer_pb2, verbio_speech_center_synthesizer_pb2_grpc

from google.protobuf.json_format import MessageToJson


class Options:
    def __init__(self):
        self.token_file = None
        self.host = ""
        self.audio_file = None
        self.language = 'en-us'
        self.secure_channel = True
        self.audio_format = 'wav'
        self.sample_rate = 16000
        self.voice: str = None
        self.text: str = None
        # self.inactivity_timeout = False
        # self.label = False
        self.client_id = None
        self.client_secret = None
    
    VOICES = {
        "en-us": ["tommy"],
        "es-419": ["miguel"],
        "ca": ["anna"],
        "es": ["david"],
        "pt-br": ["bel"]
    }

    def check(self):
        Options.__check_voice(self.voice, self.language)

    def __check_voice(voice: str, language: str) -> str:
        if voice not in Options.VOICES[language]:
            error_msg = "The only voices available for {lang} are: {voices}. (Used: {voice})".format(
                lang=language,
                voices=", ".join(Options.VOICES[language]),
                voice=voice
            )
            raise Exception(error_msg)

def parse_credential_args(args, options):
    if args.client_id and not args.client_secret:
        raise argparse.ArgumentError(None, "If --client-id is specified, then --client-secret must also be specified.")
    elif args.client_secret and not args.client_id:
        raise argparse.ArgumentError(None, "If --client-secret is specified, then --client-id must also be specified.")
    options.client_id = args.client_id or None
    options.client_secret = args.client_secret or None

def parse_command_line() -> Options:
    options = Options()
    parser = argparse.ArgumentParser(description='Perform speech synthesis on a given text')
    parser.add_argument('--list', '-L', help='Whether to list available voices (in general or for a specific language.)')
    parser.add_argument('--text', '-T', help='Text to synthesize to audio', required=True)
    parser.add_argument('--voice', '-v', choices=['tommy', 'miguel', 'anna', 'david', 'bel'], help='Voice to use for the synthesis', required=True)
    parser.add_argument('--sample-rate', '-s', type=int, choices=[16000], help='Output audio sample rate in Hz (default: ' + str(options.sample_rate) + ')', default=options.sample_rate)
    parser.add_argument('--format', '-f', choices=['wav', 'raw'], help='Output audio format (default: ' + options.audio_format + ')', default=options.audio_format)
    parser.add_argument('--language', '-l', choices=['en-us', 'es-pe', 'ca-es', 'pt-br', 'es-es'], help='A Language ID (default: ' + options.language + ')', default=options.language)
    parser.add_argument('--audio-file', '-a', help='Path to store the resulting audio', required=True)
    parser.add_argument('--token', '-t', help='File with the authentication token', required=True)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach (default: ' + options.host + ')', required=True)
    parser.add_argument('--not-secure', '-S', help='Do not use a secure channel. Used for internal testing.', required=False, default=True, dest='secure', action='store_false')
    # parser.add_argument('--inactivity-timeout', '-i', help='Time for stream inactivity after the initial request', required=False, default=5.0)
    # parser.add_argument('--label', help='"Label for the request', required=False, default="")
    
    credentialGroup = parser.add_argument_group('credentials', '[OPTIONAL] Client authentication credentials used to refresh the token. You can find your credentials on the dashboard at https://dashboard.speechcenter.verbio.com/access-token')
    credentialGroup.add_argument('--client-id', help='Client id for authentication. MUST be written as --client-id=CLIENT_ID')
    credentialGroup.add_argument('--client-secret', help='Client secret for authentication. MUST be written as --client-secret=CLIENT_SECRET')

    args = parser.parse_args()
    parse_credential_args(args, options)

    options.token_file = args.token
    options.host = args.host
    options.audio_file = args.audio_file
    options.language = args.language
    options.secure_channel = args.secure
    options.audio_format = args.format
    # options.inactivity_timeout = float(args.inactivity_timeout)
    # options.label = args.label
    options.text = args.text
    options.voice = args.voice

    return options


class GrpcChannelCredentials:
    def __init__(self, token):
        # Set JWT token for service access.
        self.call_credentials = grpc.access_token_call_credentials(token)
        # Set CA Certificate for SSL channel encryption.
        self.ssl_credentials = grpc.ssl_channel_credentials()

    def get_channel_credentials(self):
        return grpc.composite_channel_credentials(self.ssl_credentials, self.call_credentials)


class Audio:
    def __init__(self, options: Options):
        self.frame_rate = options.sample_rate
        self.audio_format = options.audio_format
        self.speaker = self.__get_speaker(options.voice, options.language)
        self.sample_rate = self.__get_sample_rate(options.sample_rate)
        self.format_value = self.__get_format(options.audio_format)

    FORMATS = {
        "wav": 0,  # AUDIO_FORMAT_WAV_LPCM_S16LE
        "raw": 1   # AUDIO_FORMAT_RAW_LPCM_S16LE
    }

    @staticmethod
    def __get_format(audio_format: str) -> str:
        if audio_format not in Audio.FORMATS:
            raise Exception(f"Format {audio_format} not supported.")
        return Audio.FORMATS[audio_format]

    @staticmethod
    def __get_speaker(voice: str, language: str) -> str:
        return f"{language.replace('-', '_').upper()}_{voice.upper()}"

    @staticmethod
    def __get_sample_rate(_sample_rate: int) -> int:
        if _sample_rate == 8000:
            raise Exception("Sample rate of 8000 is not implemented yet.")
        elif _sample_rate == 16000:
            return 1
        else:
            raise Exception("Sample rate given isn't among the valid options.")

    def save_audio(self, audio: bytes, filename: str):
        if self.audio_format == "raw":
            Audio.__save_audio_raw(audio, filename)
        elif self.audio_format == "wav":
            Audio.__save_audio_wav(audio, filename, self.frame_rate)
        else:
            raise Exception("Could not save resulting audio using provided format.")

    @staticmethod
    def __save_audio_wav(audio: bytes, filename: str, sample_rate: int):
        with wave.open(filename, 'wb') as f:
            # Set sample size in Bytes
            f.setsampwidth(2)
            # Set mono channel
            f.setnchannels(1)
            # Set sample rate
            f.setframerate(sample_rate)
            # Write audio data
            f.writeframesraw(audio)

    @staticmethod
    def __save_audio_raw(audio: bytes, filename: str):
        with open(filename, 'wb') as f:
            f.write(audio)


class SpeechCenterStreamingTTSClient:
    def __init__(self, executor: ThreadPoolExecutor, channel: grpc.Channel, options: Options):
        self._executor = executor
        self._channel = channel
        self._stub = verbio_speech_center_synthesizer_pb2_grpc.TextToSpeechStub(self._channel)
        self.audio = Audio(options)
        self._host = options.host
        self._language = options.language
        self._peer_responded = threading.Event()
        self.token = retrieve_token(options)
        self._secure_channel = options.secure_channel
        # self._inactivity_timer = None
        # self._inactivity_timer_timeout = options.inactivity_timeout
        self._audio_format = options.audio_format
        # self._label = options.label
        self.text = options.text

    def _close_stream_by_inactivity(self):
        logging.info("Stream inactivity detected, closing stream...")
        self._peer_responded.set()

    def _start_inactivity_timer(self, inactivity_timeout : float):
        self._inactivity_timer = Timer(inactivity_timeout, self._close_stream_by_inactivity)
        self._inactivity_timer.start()

    def _store_result(self, response):
        # Store the inference response audio into an audio file
        self.audio.save_audio(response.audio_samples, self.audio_file)
        logging.info("Stored resulting audio at %s", self.audio_file)

    def _response_watcher(
            self,
            response: verbio_speech_center_synthesizer_pb2.SynthesisResponse) -> None:
        try:
            logging.info("Running response watcher")
            json = MessageToJson(response)
            logging.info("New incoming response: '%s ...'", json[0:50].replace('\n', ''))
            self._store_result(response)

            if response.result:
                if self._inactivity_timer:
                    self._inactivity_timer.cancel()
                self._start_inactivity_timer(self._inactivity_timer_timeout)

        except Exception as e:
            logging.error("Error running response watcher: %s", str(e))
            self._peer_responded.set()
            raise

    def call(self) -> None:
        if self.text is None:
            raise Exception("Can't synthesize missing text.")
        metadata = [('authorization', "Bearer " + self.token)]
        if self._secure_channel:
            response = self._stub.SynthesizeSpeech(self.__generate_inferences(
                            text=self.text,
                            voice=self.audio.speaker,
                            sampling_rate=self.audio.sample_rate,
                            audio_format=self.audio.format_value
                        ))
        else:
            response = self._stub.SynthesizeSpeech(self.__generate_inferences(
                            text=self.text,
                            voice=self.audio.speaker,
                            sampling_rate=self.audio.sample_rate,
                            audio_format=self.audio.format_value
                        ), metadata=metadata)

        self._consumer_future = self._executor.submit(self._response_watcher, response)

    def wait_server(self) -> bool:
        logging.info("Waiting for server to respond...")
        self._peer_responded.wait(timeout=None)
        if self._consumer_future.done():
            # If the future raises, forwards the exception here
            self._consumer_future.result()

        return True
    
    @staticmethod
    def __generate_inferences(
        text: str,
        voice: str,
        sampling_rate: str,
        audio_format: str,
    ) -> verbio_speech_center_synthesizer_pb2.SynthesisRequest:
        message = verbio_speech_center_synthesizer_pb2.SynthesisRequest(
            text=text,
            voice=voice,
            sampling_rate=sampling_rate,
            format=audio_format
        )

        logging.info("Sending message SynthesisRequest")
        return message


def process_synthesis(executor: ThreadPoolExecutor, channel: grpc.Channel, options: Options) -> None:
    client = SpeechCenterStreamingTTSClient(executor, channel, options)
    client.call()
    if client.wait_server():
        logging.info("Recognition finished")
    else:
        logging.error("Recognition failed: server didn't answer")


def run_executor(command_line_options, executor, channel):
    logging.info("Running executor...")
    future = executor.submit(process_synthesis, executor, channel, command_line_options)
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
        with grpc.secure_channel(command_line_options.host, credentials=credentials.get_channel_credentials()) as channel:
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
    
