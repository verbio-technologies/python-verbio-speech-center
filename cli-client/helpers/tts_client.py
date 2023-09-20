import sys
sys.path.insert(1, '../proto/generated')

import grpc
import logging
from helpers.common import SynthesizerOptions
from helpers.audio_exporter import AudioExporter
import verbio_speech_center_synthesizer_pb2
import verbio_speech_center_synthesizer_pb2_grpc


class TTSClient:
    def __init__(self, channel: grpc.Channel, options: SynthesizerOptions, token: str):
        self._channel = channel
        self._secure_channel = options.secure_channel
        self._audio_format = AudioExporter.SUPPORTED_FORMATS[options.audio_format]
        self._audio_file = options.audio_file
        self._audio_sample_rate = options.sample_rate
        self._token = token
        self._stub = verbio_speech_center_synthesizer_pb2_grpc.TextToSpeechStub(self._channel)

    def _compose_synthesis_request(self, text: str, voice: str, audio_format: str, sampling_rate: int):
        message = verbio_speech_center_synthesizer_pb2.SynthesisRequest(
            text=text,
            voice=voice,
            sampling_rate=sampling_rate,
            format=audio_format
        )

        return message

    def synthesize(self, text: str, voice: str) -> bytes:
        logging.info("Sending synthesis request for voice: -%s-, sampling_rate: %i and text: -%s-", voice, self._audio_sample_rate, text)

        metadata = None if self._secure_channel else [('authorization', "Bearer " + self._token)]
        response, call = self._stub.SynthesizeSpeech.with_call(
            self._compose_synthesis_request(
                    text=text,
                    voice=voice,
                    sampling_rate=self._audio_sample_rate,
                    audio_format=self._audio_format
                ), metadata=metadata
            )

        logging.info("Synthesis response [status=%s]", str(call.code()))
        return response.audio_samples
