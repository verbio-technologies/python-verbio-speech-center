import sys
sys.path.insert(1, '../proto/generated')

import logging
import threading
from threading import Timer
from helpers.common import SynthesizerOptions
from concurrent.futures import ThreadPoolExecutor
from helpers.audio_exporter import AudioExporter
from helpers.common import split_text
import speechcenter.tts.text_to_speech_pb2 as text_to_speech_pb2


class TTSClient:
    def __init__(self, executor: ThreadPoolExecutor, stub, options: SynthesizerOptions, token: str):
        self._executor = executor
        self._stub = stub
        self._text = options.text
        self._text_file = options.text_file
        self._voice = options.voice
        self._audio_format = options.audio_format
        self._audio_file = options.audio_file
        self._audio_sample_rate = options.sample_rate
        self._peer_responded = threading.Event()
        self._token = token
        self._secure_channel = options.secure_channel
        self._inactivity_timer = None
        self._inactivity_timer_timeout = options.inactivity_timeout
        self._supported_sample_rates = {8000: text_to_speech_pb2.VoiceSamplingRate.VOICE_SAMPLING_RATE_8KHZ, 
            16000: text_to_speech_pb2.VoiceSamplingRate.VOICE_SAMPLING_RATE_16KHZ}
 
    def _compose_synthesis_request(self, text: str, voice: str, audio_format: str, sampling_rate: int):
        message = text_to_speech_pb2.SynthesisRequest(
            text=text,
            voice=voice,
            sampling_rate=sampling_rate,
            format=audio_format
        )

        return message
    
    def save_audio_result(self, audio_samples: bytes):
        audio_exporter = AudioExporter(self._audio_sample_rate)
        audio_exporter.save_audio(self._audio_format, audio_samples, self._audio_file)
        logging.info("Stored synthesis audio at -%s-", self._audio_file)

    def synthesize(self) -> bytes:
        logging.info("Sending synthesis request for voice: -%s-, sampling_rate: %i and text: -%s-", self._voice, self._audio_sample_rate, self._text)
        selected_audio_format = AudioExporter.SUPPORTED_FORMATS[self._audio_format]

        metadata = None if self._secure_channel else [('authorization', "Bearer " + self._token)]
        response, call = self._stub.SynthesizeSpeech.with_call(
            self._compose_synthesis_request(
                text=self._text,
                voice=self._voice,
                sampling_rate=self._supported_sample_rates[self._audio_sample_rate],
                audio_format=selected_audio_format
            ), metadata=metadata
        )

        logging.info("Synthesis response [status=%s]", str(call.code()))
        logging.info("Received response with %s bytes of audio data", len(response.audio_samples))
        
        return response.audio_samples

    def _close_stream_by_inactivity(self):
        logging.info("Stream inactivity detected, closing stream...")
        self._peer_responded.set()

    def _start_inactivity_timer(self, inactivity_timeout: float):
        self._inactivity_timer = Timer(inactivity_timeout, self._close_stream_by_inactivity)
        self._inactivity_timer.start()
    
    def _response_watcher(self, response_iterator):
        try:
            audio = bytearray()
            logging.info("Running response watcher")
            for response in response_iterator:
                logging.debug("New incoming response of type: %s", type(response))
                
                if response.streaming_audio.audio_samples:
                    logging.info("StreamingAudio response received: %s bytes of audio data", len(response.streaming_audio.audio_samples))
                    audio.extend(response.streaming_audio.audio_samples)
                
                if response.end_of_utterance.data:
                    logging.info("EndOfUtterance response received. Signaling end of stream with data: %s", response.end_of_utterance.data)

                if self._inactivity_timer:
                    self._inactivity_timer.cancel()
                self._start_inactivity_timer(self._inactivity_timer_timeout)
                
            self.save_audio_result(audio)

        except Exception as e:
            logging.error("Error running response watcher: %s", str(e))
            self._peer_responded.set()
            raise
    
    def send_text(self) -> None:
        metadata = None if self._secure_channel else [('authorization', "Bearer " + self._token)]
        self.__generate_messages(
                text_file=self._text_file,
                voice=self._voice,
                sample_rate=self._audio_sample_rate),
        response_iterator = self._stub.StreamingSynthesizeSpeech(self.__message_iterator(), metadata=metadata)
        self._consumer_future = self._executor.submit(self._response_watcher, response_iterator)

    def wait_for_response(self) -> bool:
        logging.info("Waiting for server to respond...")
        self._peer_responded.wait(timeout=None)
        if self._consumer_future.done():
            self._consumer_future.result()

        return True

    def __message_iterator(self):
        for message_type, message in self._messages:
            logging.info("Sending streaming message " + message_type)
            yield message
        logging.info("All audio messages sent")

    def __generate_messages(
        self, 
        text_file: str, 
        voice: str,
        sample_rate: int, 
    ):
        synthesis_config = text_to_speech_pb2.SynthesisConfig(
            voice=voice,
            sampling_rate=self._supported_sample_rates[sample_rate],
        )

        self._messages = [
            ("config",
                text_to_speech_pb2.StreamingSynthesisRequest(
                    config=synthesis_config
                )
             ),
        ]

        for line in split_text(text_file):
            self._messages.append(("text", text_to_speech_pb2.StreamingSynthesisRequest(text=line)))

        end_of_utterance = text_to_speech_pb2.EndOfUtterance(data="EndOfUtterance")

        self._messages.append(("end_of_utterance", text_to_speech_pb2.StreamingSynthesisRequest(end_of_utterance=end_of_utterance)))
