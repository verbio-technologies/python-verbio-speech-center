import unittest
from mock import patch
import logging
import random
import string
import tempfile
import numpy as np
import argparse

from asr4_streaming.recognizer import Duration
from asr4_streaming.recognizer import RecognizerService
from asr4_streaming.recognizer import StreamingRecognizeRequest
from asr4_streaming.recognizer import RecognitionConfig
from asr4_streaming.recognizer import RecognitionParameters
from asr4_streaming.recognizer import RecognitionResource
from asr4_streaming.recognizer import StreamingRecognizeResponse
from asr4_streaming.recognizer import StreamingRecognitionResult
from asr4_streaming.recognizer_v1.service import TranscriptionResult

from asr4_engine.data_classes import Signal, Segment, Transcription
from asr4_engine.data_classes.transcription import WordTiming
from asr4.engines.wav2vec.v1.engine_types import Language

import os

from typing import Any, Dict, List, Optional, Union

DEFAULT_ENGLISH_MESSAGE: str = "hello i am up and running received a message from you"
DEFAULT_SPANISH_MESSAGE: str = (
    "hola estoy levantado y en marcha y he recibido un mensaje tuyo"
)
FORMATTED_SPANISH_MESSAGE: str = (
    "Hola. Estoy levantado y en marcha y he recibido un mensaje tuyo."
)
DEFAULT_PORTUGUESE_MESSAGE: str = "ola estou de pe recebi uma mensagem sua"


class MockArguments(argparse.Namespace):
    def __init__(self, language):
        super().__init__()
        self.config = "asr4_config.toml"
        self.language = language


class MockEngine:
    def __init__(self, _path_or_bytes: Union[str, bytes], language) -> None:
        self.logger = logging.getLogger("TEST")
        self._message = {
            "en-US": DEFAULT_ENGLISH_MESSAGE,
            "es": DEFAULT_SPANISH_MESSAGE,
            "pt-BR": DEFAULT_PORTUGUESE_MESSAGE,
        }.get(language, DEFAULT_ENGLISH_MESSAGE)
        self.language = language

    def recognize(
        self,
        input: Signal,
        **kwargs,
    ) -> Transcription:
        return Transcription(
            text=self._message,
            segments=self._generateDefaultSegmentsArray(self._message),
            language=self.language,
        )

    def _generateDefaultSegmentsArray(self, defaultMessage: List[str]) -> np.ndarray:
        segments = []
        for i, token in enumerate(defaultMessage.split(" ")):
            segments.append(
                Segment(
                    id=i,
                    start_index=i,
                    end_index=i + 1,
                    seek=0,
                    start=i,
                    end=i + 1,
                    text=token,
                    tokens=[0],
                    temperature=random.uniform(0.0, 1.0),
                    avg_logprob=random.uniform(0.0, 1.0),
                    compression_ratio=random.uniform(0.0, 1.0),
                    no_speech_prob=random.uniform(0.0, 1.0),
                    words=[
                        WordTiming(
                            word=token,
                            start=i,
                            end=i + 1,
                            probability=1.0,
                        )
                    ],
                )
            )
        return segments


class TestRecognizerService(unittest.TestCase):
    def initializeEngine(self, config, language):
        return MockEngine(
            config,
            language,
        )

    def testInvalidStreamingRecognizeRequestEmpty(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = StreamingRecognizeRequest()
            with self.assertRaises(ValueError):
                service.eventSource(request.config, request.audio)

    def testInvalidStreamingRecognizeRequestAudio(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = StreamingRecognizeRequest(audio=b"SOMETHING")
            with self.assertRaises(ValueError):
                service.eventSource(RecognitionConfig(), request.audio)

    def testInvalidStreamingRecognizeRequestResource(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = StreamingRecognizeRequest(
                config=RecognitionConfig(resource=RecognitionResource(topic="GENERIC"))
            )
            with self.assertRaises(ValueError):
                service.eventSource(request.config, bytes())

    def testInvalidStreamingRecognizeRequestLanguage(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = StreamingRecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(language="en-US"),
                )
            )
            with self.assertRaises(ValueError):
                service.eventSource(request.config, bytes())

    def testInvalidStreamingRecognizeRequestSampleRate(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = StreamingRecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(sample_rate_hz=4000),
                )
            )
            with self.assertRaises(ValueError):
                service.eventSource(request.config, bytes())

    def testInvalidStreamingRecognizeRequestParameters(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = StreamingRecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000, audio_encoding="PCM"
                    ),
                )
            )
            with self.assertRaises(ValueError):
                service.eventSource(request.config, bytes())

    def testInvalidStreamingRecognizeRequestAudioEncodingValue(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = StreamingRecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000, audio_encoding="PCM"
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                )
            )
            with self.assertRaises(ValueError):
                service.eventSource(request.config, bytes())

    def testInvalidStreamingRecognizeRequestConfig(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = StreamingRecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000, audio_encoding="PCM"
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                )
            )
            with self.assertRaises(ValueError):
                service.eventSource(request.config, bytes())

    def testInvalidStreamingRecognizeRequestHandle(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._language = "en-US"
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = StreamingRecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(),
                )
            )
            with self.assertRaises(ValueError):
                service.eventHandle(request.config, bytes())

    def testRecognizeRequestSink(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            response = TranscriptionResult(
                transcription="hello world",
                score=1.0,
                words=[
                    WordTiming(word="hello", start=1.0, end=1.5, probability=1.0),
                    WordTiming(word="world", start=1.8, end=2.6, probability=1.0),
                ],
            )
            result = {
                "alternatives": [
                    {
                        "transcript": "hello world",
                        "confidence": 1.0,
                        "words": [
                            {
                                "start_time": {"seconds": 1},
                                "end_time": {"seconds": 1, "nanos": 500000000},
                                "word": "hello",
                                "confidence": 1.0,
                            },
                            {
                                "start_time": {"seconds": 1, "nanos": 800000000},
                                "end_time": {"seconds": 2, "nanos": 600000000},
                                "word": "world",
                                "confidence": 1.0,
                            },
                        ],
                    }
                ],
                "duration": {},
                "end_time": {"seconds": 0, "nanos": 0},
            }
            self.assertEqual(service.eventSink(response), RecognizeResponse(**result))

    def testRecognizeRequestSinkNoFrames(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            response = TranscriptionResult(
                transcription="",
                score=1.0,
                words=[],
            )
            result = {
                "alternatives": [
                    {
                        "transcript": "",
                        "confidence": 1.0,
                        "words": [],
                    }
                ],
                "duration": {},
                "end_time": {"seconds": 0, "nanos": 0},
            }
            self.assertEqual(service.eventSink(response), RecognizeResponse(**result))

    def testResponseParameters(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            transcription = "".join(
                random.choices(string.ascii_letters + string.digits, k=16)
            )
            response = service.eventSink(
                TranscriptionResult(
                    transcription=transcription,
                    score=1.0,
                    words=[
                        WordTiming(
                            word=transcription, start=1.0, end=1.5, probability=1.0
                        )
                    ],
                )
            )
            self.assertEqual(len(response.alternatives), 1)
            self.assertEqual(response.alternatives[0].transcript, transcription)
            self.assertEqual(response.alternatives[0].confidence, 1.0)
            self.assertEqual(len(response.alternatives[0].words), 1)
            self.assertEqual(response.alternatives[0].words[0].start_time.seconds, 1)
            self.assertEqual(response.alternatives[0].words[0].end_time.seconds, 1)
            self.assertEqual(
                response.alternatives[0].words[0].end_time.nanos, 500000000
            )

    def testStreamingResponseParameters(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            transcription = "".join(
                random.choices(string.ascii_letters + string.digits, k=16)
            )
            innerRecognizeResponse = service.eventSink(
                TranscriptionResult(
                    transcription=transcription,
                    score=1.0,
                    words=[
                        WordTiming(
                            word=transcription, start=1.0, end=1.5, probability=1.0
                        )
                    ],
                ),
                Duration(seconds=1, nanos=0),
            )
            streamingResponse = StreamingRecognizeResponse(
                results=StreamingRecognitionResult(
                    alternatives=innerRecognizeResponse.alternatives,
                )
            )
            self.assertEqual(len(streamingResponse.results.alternatives), 1)
            self.assertEqual(
                streamingResponse.results.alternatives[0].transcript, transcription
            )
            self.assertEqual(streamingResponse.results.alternatives[0].confidence, 1.0)