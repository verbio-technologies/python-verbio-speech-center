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
from asr4_streaming.recognizer import RecognizeRequest
from asr4_streaming.recognizer import StreamingRecognizeRequest
from asr4_streaming.recognizer import RecognitionConfig
from asr4_streaming.recognizer import RecognitionParameters
from asr4_streaming.recognizer import RecognitionResource
from asr4_streaming.recognizer import RecognizeResponse
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

    def sendAudioChunk(
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

    def testInvalidAudio(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000, audio_encoding="PCM"
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
                audio=b"",
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

    def testInvalidTopic(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000, audio_encoding="PCM"
                    ),
                    resource=RecognitionResource(topic=-1),
                ),
                audio=b"SOMETHING",
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

    def testInvalidAudioEncoding(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000, audio_encoding=2
                    ),
                    resource=RecognitionResource(topic=-1),
                ),
                audio=b"SOMETHING",
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

    def testInvalidLanguage(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(language="", sample_rate_hz=16000),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
                audio=b"SOMETHING",
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="INVALID", sample_rate_hz=16000
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
                audio=b"SOMETHING",
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

    def testInvalidSampleRate(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16001
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
                audio=b"SOMETHING",
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=8001
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
                audio=b"SOMETHING",
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

    def testInvalidRecognizeRequestEmpty(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest()
            with self.assertRaises(ValueError):
                service.eventSource(request)

    def testInvalidStreamingRecognizeRequestEmpty(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = StreamingRecognizeRequest()
            with self.assertRaises(ValueError):
                service.eventSource(request)

    def testInvalidRecognizeRequestAudio(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(audio=b"SOMETHING")
            with self.assertRaises(ValueError):
                service.eventSource(request)

    def testInvalidStreamingRecognizeRequestAudio(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = StreamingRecognizeRequest(audio=b"SOMETHING")
            with self.assertRaises(ValueError):
                service.eventSource(request)

    def testInvalidRecognizeRequestResource(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(resource=RecognitionResource(topic="GENERIC"))
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

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
                service.eventSource(request)

    def testInvalidRecognizeRequestLanguage(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(language="en-US"),
                )
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

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
                service.eventSource(request)

    def testInvalidRecognizeRequestAudioEncoding(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(audio_encoding="PCM"),
                )
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

    def testInvalidRecognizeRequestSampleRate(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(sample_rate_hz=4000),
                )
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

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
                service.eventSource(request)

    def testInvalidRecognizeRequestParameters(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000, audio_encoding="PCM"
                    ),
                )
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

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
                service.eventSource(request)

    def testInvalidRecognizeRequestAudioEncodingValue(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000, audio_encoding="PCM"
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                )
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

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
                service.eventSource(request)

    def testInvalidRecognizeRequestConfig(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000, audio_encoding=1
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
                audio=b"SOMETHING",
            )
            with self.assertRaises(ValueError):
                service.eventSource(request)

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
                service.eventSource(request)

    def testRecognizeRequestSampleRate16000(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=16000, audio_encoding="PCM"
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
                audio=b"SOMETHING",
            )
            self.assertFalse(service.eventSource(request))

    def testRecognizeRequestSampleRate8000(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US", sample_rate_hz=8000, audio_encoding="PCM"
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
                audio=b"SOMETHING",
            )
            self.assertFalse(service.eventSource(request))

    def testInvalidRecognizeRequestHandle(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._language = "en-US"
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(),
                )
            )
            with self.assertRaises(ValueError):
                service.eventHandle(request)

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
                service.eventHandle(request)

    def testRecognizeRequestHandleEnUs(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._language = Language.EN_US
            service._handler = self.initializeEngine(
                arguments.config, arguments.language
            )
            service._languageCode = "en-US"
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="en-US",
                        sample_rate_hz=8000,
                        audio_encoding="PCM",
                        enable_formatting=False,
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
                audio=b"0000",
            )
            self.assertEqual(
                service.eventHandle(request).transcription,
                DEFAULT_ENGLISH_MESSAGE,
            )

    def testRecognizeRequestHandleEs(self):
        arguments = MockArguments(language="es")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._language = Language.ES
            service._handler = self.initializeEngine(
                arguments.config, arguments.language
            )
            service._languageCode = "es"
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="es",
                        sample_rate_hz=8000,
                        audio_encoding="PCM",
                        enable_formatting=False,
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
                audio=b"0000",
            )
            self.assertEqual(
                service.eventHandle(request).transcription,
                DEFAULT_SPANISH_MESSAGE,
            )

    def testRecognizeRequestHandlePtBr(self):
        arguments = MockArguments(language="pt-BR")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._language = Language.PT_BR
            service._handler = self.initializeEngine(
                arguments.config, arguments.language
            )
            service._languageCode = "pt-BR"
            request = RecognizeRequest(
                config=RecognitionConfig(
                    parameters=RecognitionParameters(
                        language="pt-BR",
                        sample_rate_hz=8000,
                        audio_encoding="PCM",
                        enable_formatting=False,
                    ),
                    resource=RecognitionResource(topic="GENERIC"),
                ),
                audio=b"0000",
            )
            self.assertEqual(
                service.eventHandle(request).transcription,
                DEFAULT_PORTUGUESE_MESSAGE,
            )

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

    def testAudioDuration(self):
        arguments = MockArguments(language="en-US")
        with patch.object(RecognizerService, "__init__", lambda x, y: None):
            service = RecognizerService(arguments)
            service._engine = self.initializeEngine(
                arguments.config, arguments.language
            )
            config16 = RecognitionConfig(
                parameters=RecognitionParameters(sample_rate_hz=16000)
            )
            config1 = RecognitionConfig(
                parameters=RecognitionParameters(sample_rate_hz=1)
            )

            request = RecognizeRequest(audio=b"", config=config16)
            duration = service.calculateAudioDuration(request)
            self.assertEqual(duration.seconds, 0)
            self.assertEqual(duration.nanos, 0)

            request = RecognizeRequest(audio=b"0124", config=config16)
            duration = service.calculateAudioDuration(request)
            self.assertEqual(duration.seconds, 0)
            self.assertEqual(duration.nanos, 125000)

            request = RecognizeRequest(audio=b"12345678901234567890", config=config16)
            duration = service.calculateAudioDuration(request)
            self.assertEqual(duration.seconds, 0)
            self.assertEqual(duration.nanos, 625000)

            request = RecognizeRequest(audio=b"0124", config=config1)
            duration = service.calculateAudioDuration(request)
            self.assertEqual(duration.seconds, 2)
            self.assertEqual(duration.nanos, 0)

            with self.assertRaises(ZeroDivisionError):
                request = RecognizeRequest(
                    audio=b"0124",
                    config=RecognitionConfig(
                        parameters=RecognitionParameters(sample_rate_hz=0)
                    ),
                )
                service.calculateAudioDuration(request)
