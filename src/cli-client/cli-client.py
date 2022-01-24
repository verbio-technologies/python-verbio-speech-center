#!/usr/bin/env python3

import os
import string
import wave
import logging
import argparse
from typing import Iterable

import grpc
import csr_grpc_gateway_pb2
import csr_grpc_gateway_pb2_grpc


class Options:
    def __init__(self):
        self.token = None
        self.host = "https://www.speech-center.verbio.com:2424"
        self.audio_file = None
        self.grammar_file = None
        self.topic = None

    def check(self):
        if self.grammar_file is not None and self.topic is not None:
            raise Exception("You  must provide either a grammar or a topic but not both.")
        if self.grammar_file is None and self.topic is None:
            raise Exception("You must provide a least a topic or a grammar, but not both.")


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
        self.grammar = None
        self.audio = self.__read_audio_file(options.audio_file)
        if options.grammar_file:
            self.grammar = self.__read_grammar_from_file(options.grammar_file)

    @staticmethod
    def __read_grammar_from_file(grammar_file: str) -> str:
        with open(grammar_file, "r") as file:
            return file.read()

    @staticmethod
    def __read_audio_file(audio_file: str) -> bytes:
        with open(audio_file, "rb") as wav_file:
            wav_data = wave.open(wav_file)
            audio_data = wav_data.readframes(wav_data.getnframes())
            return audio_data


class SpeechCenterClient:
    def __init__(self, options: Options):
        options.check()
        self.credentials = Credentials(options.token)
        self.resources = Resources(options)
        self.host = options.host
        self.topic = options.topic

    def run(self):
        logging.info("Running CSR inference example...")
        # Open connection to grpc channel to provided host.
        with grpc.secure_channel(self.host, credentials=self.credentials,
                                 options=(('grpc.ssl_target_name_override', 'speech-center.verbio.com'),)) as channel:
            # Instantiate a speech_recognizer to manage grpc calls to backend.
            speech_recognizer = csr_grpc_gateway_pb2_grpc.SpeechRecognizerStub(channel)
            try:
                # Send inference requests with for the audio.
                # if a grammar is provided:
                if self.resources.grammar:
                    response, call = speech_recognizer.RecognizeStream.with_call(
                        self.__generate_inferences(grammar=self.resources.grammar, wav_audio=self.resources.audio))
                # or if a topic is provided:
                else:
                    response, call = speech_recognizer.RecognizeStream.with_call(
                        self.__generate_inferences(topic=self.topic, wav_audio=self.resources.audio))

                # Print out inference response and call status
                logging.info("Inference response: '%s'", response.text)
                logging.info("Inference call status: ")
                logging.info(call)

            except Exception as ex:
                logging.critical(ex)

    @staticmethod
    def __generate_inferences(self, wav_audio: bytes, topic: str = "", grammar: str = "") -> Iterable[
        csr_grpc_gateway_pb2.RecognitionRequest]:
        """
        Inferences always start with a grammar/topic and a language, then audio is passed in a second message
        """
        if len(topic):
            var_resource = csr_grpc_gateway_pb2.RecognitionResource(topic=topic)
        elif len(grammar):
            var_resource = csr_grpc_gateway_pb2.RecognitionResource(inline_grammar=grammar)
        else:
            raise Exception("Grammar or topic must be declared in order to perform the recognition")

        messages = [
            csr_grpc_gateway_pb2.RecognitionRequest(init=csr_grpc_gateway_pb2.RecognitionInit(
                parameters=csr_grpc_gateway_pb2.RecognitionParameters(language="en-US"), resource=var_resource)),
            csr_grpc_gateway_pb2.RecognitionRequest(audio=wav_audio),
        ]
        for msg in messages:
            logging.info("Sending message...")
            yield msg


def parse_command_line() -> Options:
    options = Options()
    parser = argparse.ArgumentParser(description='Perform speech recognition on an audio file')
    parser.add_argument('--audiofile', '-a', help='Path to a .wav audio in 8kHz and PCM16 encoding', required=True)
    argGroup = parser.add_mutually_exclusive_group(required=True)
    argGroup.add_argument('--grammar', '-g', help='Path to a file containing an ABNF grammar')
    argGroup.add_argument('--topic', '-T', choices=['GENERIC', 'TELCO', 'BANKING'], help='A valid topic')
    parser.add_argument('--token', '-t', help='A string with the authentication token', required=True)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach',
                        default='speechcenter.verbio.com:2424')
    args = parser.parse_args()
    options.token = args.token
    options.host = args.host
    options.audio_file = args.audiofile
    options.grammar_file = args.grammar
    options.topic = args.topic

    return options


if __name__ == '__main__':
    # Setup minimal logger and run example.
    logging.basicConfig(level=logging.INFO)

    SpeechCenterClient(parse_command_line()).run()
