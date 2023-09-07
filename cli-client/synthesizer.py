#!/usr/bin/env python3
# Used to make sure python find proto files
import sys

sys.path.insert(1, '../proto/generated')

import argparse
import logging
import wave
import grpc
from speechcenterauth import SpeechCenterCredentials
import verbio_speech_center_synthesizer_pb2, verbio_speech_center_synthesizer_pb2_grpc

from google.protobuf.json_format import MessageToJson


class Options:
    def __init__(self):
        self.list_request = False
        self.token_file = None
        self.host = ""
        self.audio_file = None
        self.language = None
        self.secure_channel = True
        self.audio_format = 'wav'
        self.sample_rate = 16000
        self.voice: str = None
        self.text: str = None
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
        if self.language is None:
            raise Exception("Missing language.")
        Options.__check_voice(self.voice, self.language)

    def __check_voice(voice: str, language: str) -> str:
        if voice not in Options.VOICES[language]:
            error_msg = "The only voices available for {lang} are: {voices}. (Used: {voice})".format(
                lang=language,
                voices=", ".join(Options.VOICES[language]),
                voice=voice
            )
            raise Exception(error_msg)

    def is_list_request(self):
        if not self.list_request:
            return False
        else:
            not_needed = []
            if self.audio_format is not None:
                not_needed.append("format")
            if self.sample_rate is not None:
                not_needed.append("sample-rate")
            if self.audio_file is not None:
                not_needed.append("audio-file")
            warning_msg = "For a Listing Voices request, the following fields are not necessary: {}.".format(
                ", ".join(not_needed)
            )
            logging.warning(warning_msg)
            return True


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
    parser.add_argument('--list', '-L', help='Request a list available voices (in general or for a specific language). Note: This is mutually exclusive with text to speech synthesis.',
                        required=False, default=False, dest='list', action='store_true')
    parser.add_argument('--text', '-T', help='Text to synthesize to audio', required=True)
    parser.add_argument('--voice', '-v', choices=['tommy', 'miguel', 'anna', 'david', 'bel'], help='Voice to use for the synthesis', required=True)
    parser.add_argument('--sample-rate', '-s', type=int, choices=[16000], help='Output audio sample rate in Hz (default: ' + str(options.sample_rate) + ')', default=options.sample_rate)
    parser.add_argument('--format', '-f', choices=['wav', 'raw'], help='Output audio format (default: ' + options.audio_format + ')', default=options.audio_format)
    parser.add_argument('--language', '-l', choices=['en-us', 'es-419', 'ca', 'pt', 'es'], help='A Language ID')
    parser.add_argument('--audio-file', '-a', help='Path to store the resulting audio', required=True)
    parser.add_argument('--token', '-t', help='File with the authentication token', required=True)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach (default: ' + options.host + ')', required=True)
    parser.add_argument('--not-secure', '-S', help='Do not use a secure channel. Used for internal testing.', required=False, default=True, dest='secure', action='store_false')
    
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
    options.text = args.text
    options.voice = args.voice
    options.list_request = args.list

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


class SpeechCenterGRPCClient:
    def __init__(self, channel: grpc.Channel, options: Options):
        self._channel = channel
        self._stub = verbio_speech_center_synthesizer_pb2_grpc.TextToSpeechStub(self._channel)
        self.audio = Audio(options)
        self._host = options.host
        self._language = options.language
        self.token = retrieve_token(options)
        self._secure_channel = options.secure_channel
        self._audio_format = options.audio_format
        self.text = options.text
    
    def log_response(call):
        # Print out inference response and call status
        logging.info("Synthesis response [status=%s]", str(call.code()))
    
    def print_response(response):
        json = MessageToJson(response)
        logging.info("New incoming response: '%s'", json)


class SpeechCenterListVoicesClient(SpeechCenterGRPCClient):
    @staticmethod
    def __generate_inferences(
        language: str = None,
    ) -> verbio_speech_center_synthesizer_pb2.ListVoicesRequest:
        if language is not None:
            message = verbio_speech_center_synthesizer_pb2.ListVoicesRequest(
                language=language,
            )
        else:
            message = verbio_speech_center_synthesizer_pb2.ListVoicesRequest()

        logging.info("Sending message ListVoicesRequest")
        return message

    def run(self):
        if self.text is None:
            raise Exception("Can't synthesize missing text.")
        metadata = [('authorization', "Bearer " + self.token)]
        if self._secure_channel:
            response, call = self._stub.ListVoices.with_call(
                self.__generate_inferences(
                        language=self._language
                    )
                )
        else:
            response, call = self._stub.ListVoices.with_call(
                self.__generate_inferences(
                        language=self._language
                    ), metadata=metadata
                )

        self.log_response(call)
        self.print_response(response)


class SpeechCenterTTSClient(SpeechCenterGRPCClient):
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

    def run(self):
        if self.text is None:
            raise Exception("Can't synthesize missing text.")
        metadata = [('authorization', "Bearer " + self.token)]
        if self._secure_channel:
            response, call = self._stub.SynthesizeSpeech.with_call(
                self.__generate_inferences(
                        text=self.text,
                        voice=self.audio.speaker,
                        sampling_rate=self.audio.sample_rate,
                        audio_format=self.audio.format_value
                    )
                )
        else:
            response, call = self._stub.SynthesizeSpeech.with_call(
                self.__generate_inferences(
                        text=self.text,
                        voice=self.audio.speaker,
                        sampling_rate=self.audio.sample_rate,
                        audio_format=self.audio.format_value
                    ), metadata=metadata
                )

        self.log_response(call)
        self.audio.save_audio(response.audio_samples, self.audio_file)
        logging.info("Stored resulting audio at %s", self.audio_file)


def process_request(channel: grpc.Channel, options: Options) -> None:
    if options.is_list_request():
        process_list_request(channel, options)
    else:
        process_synthesis(channel, options)

def process_list_request(channel: grpc.Channel, options: Options) -> None:
    client = SpeechCenterListVoicesClient(channel, options)
    client.run()

def process_synthesis(channel: grpc.Channel, options: Options) -> None:
    client = SpeechCenterTTSClient(channel, options)
    client.run()

def retrieve_token(options: Options):
    if options.client_id:
        return SpeechCenterCredentials.get_refreshed_token(options.client_id, options.client_secret, options.token_file)
    else:
        return SpeechCenterCredentials.read_token(token_file=options.token_file)

def run(command_line_options: Options):
    logging.info("Connecting to %s", command_line_options.host)

    if command_line_options.secure_channel:
        token = retrieve_token(command_line_options)
        credentials = GrpcChannelCredentials(token)
        with grpc.secure_channel(command_line_options.host, credentials=credentials.get_channel_credentials()) as channel:
            process_request(channel, command_line_options)
    else:
        with grpc.insecure_channel(command_line_options.host) as channel:
            process_request(channel, command_line_options)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]:%(message)s')
    logging.info("Running speechcenter streaming channel...")
    command_line_options = parse_command_line()
    command_line_options.check()
    run(command_line_options)
