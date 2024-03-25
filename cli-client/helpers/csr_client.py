import sys
sys.path.insert(1, '../proto/generated')

import logging
import threading
from threading import Timer
from helpers.common import VerbioGrammar, RecognizerOptions
from concurrent.futures import ThreadPoolExecutor
from google.protobuf.json_format import MessageToJson
from helpers.common import split_audio
import recognition_streaming_request_pb2


class CSRClient:
    def __init__(self, executor: ThreadPoolExecutor, stub, options: RecognizerOptions, audio_resource, token: str):
        self._executor = executor
        self._stub = stub
        self._resources = audio_resource
        self._host = options.host
        self._topic = options.topic
        self._grammar = options.grammar
        self._language = options.language
        self._peer_responded = threading.Event()
        self._token = token
        self._secure_channel = options.secure_channel
        self._inactivity_timer = None
        self._inactivity_timer_timeout = options.inactivity_timeout
        self._asr_version = options.asr_version
        self._formatting = options.formatting
        self._diarization = options.diarization
        self._label = options.label
        self._messages = None

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
                print('\t"transcript": "%s",\n\t"confidence": "%f",\n\t"duration": "%f"' % (alternative.transcript, alternative.confidence, duration))

    def _response_watcher(self, response_iterator):
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

            self._inactivity_timer.cancel()
            self._peer_responded.set()

        except Exception as e:
            logging.error("Error running response watcher: %s", str(e))
            self._peer_responded.set()
            raise

    def send_audio(self) -> None:
        metadata = None if self._secure_channel else [('authorization', "Bearer " + self._token)]
        self.__generate_messages(
                topic=self._topic,
                grammar=self._grammar,
                asr_version=self._asr_version,
                wav_audio=self._resources.audio,
                language=self._language,
                sample_rate=self._resources.sample_rate,
                formatting=self._formatting,
                diarization=self._diarization,
                label=self._label)
        response_iterator = self._stub.StreamingRecognize(self.__message_iterator(), metadata=metadata)
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

    def __generate_grammar_resource(self, grammar):
        if grammar.type == VerbioGrammar.INLINE:
            return recognition_streaming_request_pb2.GrammarResource(inline_grammar=grammar.content)
        elif grammar.type == VerbioGrammar.URI:
            return recognition_streaming_request_pb2.GrammarResource(grammar_uri=grammar.content)
        elif grammar.type == VerbioGrammar.COMPILED:
            with open(grammar.content, "rb") as grammar_file:
                compiled_grammar = grammar_file.read()
                return recognition_streaming_request_pb2.GrammarResource(compiled_grammar=compiled_grammar)

        raise Exception("Type of grammar not recognized.")

    def __generate_recognition_resource(self, topic, grammar):
        if grammar:
            grammar_resource = self.__generate_grammar_resource(grammar)
            return recognition_streaming_request_pb2.RecognitionResource(grammar=grammar_resource)
        else:
            return recognition_streaming_request_pb2.RecognitionResource(topic=topic)

    def __generate_messages(self,
                            wav_audio: bytes,
                            asr_version: str,
                            topic: str = "",
                            grammar: str = "",
                            language: str = "",
                            sample_rate: int = 16000,
                            diarization=False,
                            formatting=False,
                            label: str = ""):

        resource = self.__generate_recognition_resource(topic, grammar)
        asr_versions = {"V1": 0, "V2": 1}
        selected_asr_version = asr_versions[asr_version]

        recognition_config = recognition_streaming_request_pb2.RecognitionConfig(
                        parameters=recognition_streaming_request_pb2.RecognitionParameters(
                            language=language,
                            pcm=recognition_streaming_request_pb2.PCM(sample_rate_hz=sample_rate),
                            enable_formatting=formatting,
                            enable_diarization=diarization
                        ),
                        resource=resource,
                        label=[label],
                        version=selected_asr_version
                    )

        self._messages = [
            ("config",
                recognition_streaming_request_pb2.RecognitionStreamingRequest(
                    config=recognition_config
                )
             ),
        ]

        for chunk in split_audio(wav_audio):
            logging.debug("Appending chunk as message: " + repr(chunk)[0:20] + "...")
            self._messages.append(("audio", recognition_streaming_request_pb2.RecognitionStreamingRequest(audio=chunk)))
