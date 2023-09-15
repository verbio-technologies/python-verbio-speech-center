#!/usr/bin/env python3
# Used to make sure python finds proto files
import sys
sys.path.insert(1, '../proto/generated')

import logging
import wave
import grpc
from common import SynthesizerOptions, parse_tts_command_line
from speechcenterauth import SpeechCenterCredentials
import verbio_speech_center_synthesizer_pb2, verbio_speech_center_synthesizer_pb2_grpc
from google.protobuf.json_format import MessageToJson


class GrpcChannelCredentials:
    def __init__(self, token):
        # Set JWT token for service access.
        self.call_credentials = grpc.access_token_call_credentials(token)
        # Set CA Certificate for SSL channel encryption.
        self.ssl_credentials = grpc.ssl_channel_credentials()

    def get_channel_credentials(self):
        return grpc.composite_channel_credentials(self.ssl_credentials, self.call_credentials)


class Audio:
    def __init__(self, options: SynthesizerOptions):
        self.frame_rate = options.sample_rate
        self.audio_format = options.audio_format
        self.voice = options.voice
        self.sample_rate = options.sample_rate
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
    def __init__(self, channel: grpc.Channel, options: SynthesizerOptions, token: str):
        self._channel = channel
        self._stub = verbio_speech_center_synthesizer_pb2_grpc.TextToSpeechStub(self._channel)
        self.audio = Audio(options)
        self._host = options.host
        self._secure_channel = options.secure_channel
        self._audio_format = options.audio_format
        self.text = options.text
        self._audio_file = options.audio_file
        self.token = token
    
    def log_response(self, call):
        # Print out inference response and call status
        logging.info("Synthesis response [status=%s]", str(call.code()))
    
    def print_response(response):
        json = MessageToJson(response)
        logging.info("New incoming response: '%s'", json)


class SpeechCenterTTSClient(SpeechCenterGRPCClient):
    @staticmethod
    def __send_synthesis_request(
        text: str,
        voice: str,
        sampling_rate: str,
        audio_format: str,
    ) -> verbio_speech_center_synthesizer_pb2.SynthesisRequest:
        logging.info("Sending message SynthesisRequest")
        message = verbio_speech_center_synthesizer_pb2.SynthesisRequest(
            text=text,
            voice=voice,
            sampling_rate=sampling_rate,
            format=audio_format
        )

        return message

    def run(self):
        if self._secure_channel:
            response, call = self._stub.SynthesizeSpeech.with_call(
                self.__send_synthesis_request(
                        text=self.text,
                        voice=self.audio.voice,
                        sampling_rate=self.audio.sample_rate,
                        audio_format=self.audio.format_value
                    )
                )
        else:
            metadata = [('authorization', "Bearer " + self.token)]
            response, call = self._stub.SynthesizeSpeech.with_call(
                self.__send_synthesis_request(
                        text=self.text,
                        voice=self.audio.voice,
                        sampling_rate=self.audio.sample_rate,
                        audio_format=self.audio.format_value
                    ), metadata=metadata
                )

        logging.info("Synthesis response [status=%s]", str(call.code()))
        self.audio.save_audio(response.audio_samples, self._audio_file)
        logging.info("Stored resulting audio at %s", self._audio_file)


def retrieve_token(options: SynthesizerOptions):
    if options.client_id:
        return SpeechCenterCredentials.get_refreshed_token(options.client_id, options.client_secret, options.token_file)
    else:
        return SpeechCenterCredentials.read_token(token_file=options.token_file)

def run(command_line_options: SynthesizerOptions):
    host = command_line_options.host
    token = retrieve_token(command_line_options)

    if command_line_options.secure_channel:
        logging.info("Connecting to %s using a secure channel...", host)
        credentials = GrpcChannelCredentials(token)
        with grpc.secure_channel(command_line_options.host, credentials=credentials.get_channel_credentials()) as channel:
            client = SpeechCenterTTSClient(channel, command_line_options, token)
            client.run()
    else:
        logging.info("Connecting to %s using a insecure channel...", host)
        with grpc.insecure_channel(command_line_options.host) as channel:
            client = SpeechCenterTTSClient(channel, command_line_options, token)
            client.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]:%(message)s')
    logging.info("Running speechcenter TTS synthesis...")
    command_line_options = parse_tts_command_line()
    run(command_line_options)
