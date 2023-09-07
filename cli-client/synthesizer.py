#!/usr/bin/env python3
# Used to make sure python find proto files
import sys

sys.path.insert(1, '../proto/generated')


import wave
import logging
import argparse

import grpc
import texttospeech_test_pb2
import texttospeech_test_pb2_grpc


class Options:
    def __init__(self):
        self.token_file = None
        self.host = 'tts.verbiospeechcenter.com'
        self.text = None
        self.voice = None
        self.sample_rate = 16000
        self.encoding = 'PCM'
        self.header = 'wav'
        self.language = 'en-us'

    def check(self):
        Options.__check_voice(self.voice, self.language)
        Options.__check_audio_format(self.header, self.encoding)

    def __check_voice(voice: str, language: str) -> str:
        if language == "en-us" and voice not in ["tommy"]:
            raise Exception("Only tommy is available for en-US.")
        elif language == "es-pe" and voice not in ["miguel"]:
            raise Exception("Only miguel is available for es-pe.")

    def __check_audio_format(header: str, encoding: str) -> str:
        if encoding == "PCM" and header not in ["wav", "raw"]:
            raise Exception("Only wav and raw headers are available for PCM audio encoding.")


class Credentials:
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
        self.header = options.header
        self.speaker = self.__get_speaker(options.voice, options.language)
        self.sample_rate = self.__get_sample_rate(options.sample_rate)
        self.audio_format = self.__get_audio_format(options.header, options.encoding)

    @staticmethod
    def __get_speaker(voice: str, language: str) -> str:
        return f"{language.replace('-', '_').upper()}_{voice.upper()}"

    @staticmethod
    def __get_sample_rate(_sample_rate: int) -> int:
        return 16000

    @staticmethod
    def __get_audio_format(header: str, _encoding: str) -> int:
        return 1

    def save_audio(self, audio: bytes, filename: str):
        if self.header == "raw":
            Audio.__save_audio_raw(audio, filename)
        elif self.header == "wav":
            Audio.__save_audio_wav(audio, filename, self.frame_rate)
        else:
            raise Exception("Could not save resulting audio using provided header.")

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

class SpeechCenterSynthesisClient:
    def __init__(self, options: Options):
        options.check()
        self.options = options
        self.audio = Audio(options)
        self.host = options.host
        self.text = options.text
        self.audio_file = options.audio_file

    def run(self):
        logging.info("Running Synthesizer inference example...")
        # Open connection to grpc channel to provided host.
        with grpc.secure_channel(self.host, credentials=grpc.ssl_channel_credentials()) as channel:
            # Instantiate a speech_synthesizer to manage grpc calls to backend.
            speech_synthesizer = texttospeech_test_pb2_grpc.TextToSpeechStub(channel)
            try:
                # Send inference requests for the text.
                response, call = speech_synthesizer.SynthesizeSpeech.with_call(
                        self.__generate_inferences(text=self.text, voice=self.options.voice, language=self.options.language))
                # Print out inference response and call status
                logging.info("Synthesis response [status=%s]", str(call.code()))
                # Store the inference response audio into an audio file
                self.audio.save_audio(response.audio_samples, self.audio_file)
                logging.info("Stored resulting audio at %s", self.audio_file)

            except Exception as ex:
                logging.critical(ex)

    @staticmethod
    def __generate_inferences(
        text: str,
        voice: str,
        language: str,
    ) -> texttospeech_test_pb2.SynthesizeSpeechRequest:
        message = texttospeech_test_pb2.SynthesizeSpeechRequest(
            text=text,
            language=language,
            speaker=voice
        )
        logging.info("Sending message SynthesisRequest")
        return message

    @staticmethod
    def __read_token(toke_file: str) -> str:
        with open(toke_file) as token_hdl:
            return ''.join(token_hdl.read().splitlines())


def parse_command_line() -> Options:
    options = Options()
    parser = argparse.ArgumentParser(description='Perform speech synthesis on a sample text')
    parser.add_argument('--text', '-T', help='Text to synthesize to audio', required=True)
    parser.add_argument('--voice', '-v', choices=['tommy', 'miguel', 'anna', 'david_es', 'bel'], help='Voice to use for the synthesis', required=True)
    parser.add_argument('--sample-rate', '-s', type=int, choices=[16000], help='Output audio sample rate in Hz (default: ' + str(options.sample_rate) + ')', default=options.sample_rate)
    parser.add_argument('--encoding', '-e', choices=['PCM'], help='Output audio encoding algorithm (default: ' + options.encoding + ' [Signed 16-bit little endian PCM])', default=options.encoding)
    parser.add_argument('--format', '-f', choices=['wav', 'raw'], help='Output audio header (default: ' + options.header + ')', default=options.header)
    parser.add_argument('--language', '-l', choices=['en-us', 'es-pe', 'ca', 'pt-br', 'es', ], help='A Language ID (default: ' + options.language + ')', default=options.language)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach (default: ' + options.host + ')',
                        default=options.host)
    parser.add_argument('--audio-file', '-a', help='Path to store the resulting audio', required=True)
    args = parser.parse_args()
    options.audio_file = args.audio_file
    options.host = args.host
    options.text = args.text
    options.voice = args.voice
    options.language = args.language
    print(args.language, options.language)
    options.sample_rate = args.sample_rate
    options.encoding = args.encoding
    options.header = args.format

    return options


if __name__ == '__main__':
    print("This module is only intended for demo purposes")
    # Setup minimal logger and run example.
    logging.basicConfig(level=logging.INFO)
    SpeechCenterSynthesisClient(parse_command_line()).run()