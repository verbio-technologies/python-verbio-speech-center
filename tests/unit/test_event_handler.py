import random
import string
import unittest
from unittest.mock import Mock
from dataclasses import dataclass
from typing import List, Optional, Union, Iterator, AsyncIterator

from asr4_streaming.recognizer import Duration
from asr4_streaming.recognizer import RecognitionConfig
from asr4_streaming.recognizer import RecognitionParameters
from asr4_streaming.recognizer import RecognitionResource
from asr4_streaming.recognizer import StreamingRecognizeRequest
from asr4_streaming.recognizer import StreamingRecognizeResponse

from asr4_streaming.recognizer_v1.types import SampleRate
from asr4_streaming.recognizer_v1.handler import EventHandler
from asr4_streaming.recognizer_v1.handler import TranscriptionResult

from asr4_engine.data_classes import Transcription
from asr4.engines.wav2vec.v1.engine_types import Language
from asr4_engine.data_classes.transcription import WordTiming

DEFAULT_ENGLISH_MESSAGE: str = "hello i am up and running received a message from you"
DEFAULT_SPANISH_MESSAGE: str = (
    "hola estoy levantado y en marcha y he recibido un mensaje tuyo"
)
FORMATTED_SPANISH_MESSAGE: str = (
    "Hola. Estoy levantado y en marcha y he recibido un mensaje tuyo."
)
DEFAULT_PORTUGUESE_MESSAGE: str = "ola estou de pe recebi uma mensagem sua"


def initializeMockEngine(mock: Mock, language: str):
    message = {
        "en-US": DEFAULT_ENGLISH_MESSAGE,
        "es": DEFAULT_SPANISH_MESSAGE,
        "pt-BR": DEFAULT_PORTUGUESE_MESSAGE,
    }.get(language, DEFAULT_ENGLISH_MESSAGE)
    t = Transcription.fromTimestamps(
        score=random.uniform(0.0, 1.0),
        words=message.split(),
        wordTimestamps=[(i, i + 1) for i in range(len(message.split()))],
        wordFrames=[[i] for i in range(len(message.split()))],
        language=language,
    )
    t.initializeSegmentsFromWords()
    mock.recognize.return_value = t
    return mock


def initializeMockContext(mock: Mock):
    async def abort(_statusCode, message):
        raise Exception(message)

    mock.abort = abort
    return mock


async def asyncStreamingRequestIterator(
    language: Optional[str] = None,
    sampleRate: Optional[int] = None,
    audioEncoding: Optional[Union[int, str]] = None,
    topic: Optional[Union[int, str]] = None,
    audio: List[bytes] = [],
) -> AsyncIterator[StreamingRecognizeRequest]:
    for message in streamingRequestIterator(
        language, sampleRate, audioEncoding, topic, audio
    ):
        yield message
    return


def streamingRequestIterator(
    language: Optional[str] = None,
    sampleRate: Optional[int] = None,
    audioEncoding: Optional[Union[int, str]] = None,
    topic: Optional[Union[int, str]] = None,
    audio: List[bytes] = [],
) -> Iterator[StreamingRecognizeRequest]:
    config = RecognitionConfig()
    if topic:
        resource = RecognitionResource(topic=topic)
        config.resource.CopyFrom(resource)
    if language or sampleRate or audioEncoding:
        parameters = RecognitionParameters(
            language=language or "",
            sample_rate_hz=sampleRate or 0,
            audio_encoding=audioEncoding or "PCM",
        )
        config.parameters.CopyFrom(parameters)
    yield StreamingRecognizeRequest(config=config)
    yield from map(lambda x: StreamingRecognizeRequest(audio=x), audio)
    return


class TestRecognizerServiceUtils(unittest.TestCase):
    def testCalculateAverageScore(self):
        def mockSegments():
            @dataclass
            class SegmentMock:
                avg_logprob: float

            return [
                SegmentMock(avg_logprob=0.9740297068720278),
                SegmentMock(avg_logprob=0.4466984412249397),
                SegmentMock(avg_logprob=0.24860173759730994),
            ]

        self.assertEqual(EventHandler._EventHandler__calculateAverageScore([]), 0.0)
        self.assertEqual(
            EventHandler._EventHandler__calculateAverageScore(mockSegments()),
            0.5564432952314259,
        )


class TestEventHandler(unittest.IsolatedAsyncioTestCase):
    async def testEmptyRequest(self):
        async def requestIterator() -> AsyncIterator[StreamingRecognizeRequest]:
            yield StreamingRecognizeRequest()
            return

        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        with self.assertRaises(Exception) as context:
            async for request in requestIterator():
                await handler.source(request)
        self.assertEqual(str(context.exception), "Empty request")

    async def testInvalidAudio(self):
        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b""],
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
        self.assertEqual(str(context.exception), "Empty value for audio")

    async def testInvalidTopic(self):
        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic=-1,
            audio=[b"SOMETHING"],
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
        self.assertEqual(
            str(context.exception), "Invalid value '-1' for topic resource"
        )

    async def testInvalidAudioEncoding(self):
        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding=2,
            topic="GENERIC",
            audio=[b"SOMETHING"],
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
        self.assertEqual(
            str(context.exception), "Invalid value '2' for audio_encoding parameter"
        )

    async def testInvalidSampleRate(self):
        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="en-US", sampleRate=16001, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
        self.assertEqual(
            str(context.exception), "Invalid value '16001' for sample_rate_hz parameter"
        )

    async def testInvalidLanguage(self):
        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="", sampleRate=16000, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
        self.assertEqual(
            str(context.exception), "Invalid value '' for language parameter"
        )

        requestIterator = asyncStreamingRequestIterator(
            language="INVALID", sampleRate=16000, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
        self.assertEqual(
            str(context.exception), "Invalid value 'INVALID' for language parameter"
        )

    async def testMissingConfig(self):
        async def requestIterator() -> AsyncIterator[StreamingRecognizeRequest]:
            yield StreamingRecognizeRequest(audio=b"SOMETHING")
            return

        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        with self.assertRaises(Exception) as context:
            async for request in requestIterator():
                await handler.source(request)
            await handler.handle()
        self.assertEqual(
            str(context.exception),
            "A request containing RecognitionConfig must be sent first",
        )

    async def testMissingConfigParameters(self):
        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(topic="GENERIC")
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    async def testMissingConfigParametersExceptLanguage(self):
        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(language="en-US")
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    async def testMissingConfigParametersExceptEncoding(self):
        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(audioEncoding="PCM")
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    async def testMissingConfigParametersExceptSampleRate(self):
        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(sampleRate=8000)
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
        self.assertEqual(
            str(context.exception), "Invalid value '' for language parameter"
        )

    async def testMissingAudio(self):
        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="en-US", sampleRate=16000
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
            await handler.handle()
        self.assertEqual(str(context.exception), "Audio was never received")

    async def testIncorrectLanguage(self):
        mockContext = initializeMockContext(Mock())
        handler = EventHandler(Language.ES, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.source(request)
            await handler.handle()
        self.assertEqual(
            str(context.exception), "Invalid language 'en-US'. Only 'es' is supported."
        )

    async def testRecognitionWithAllSampleRates(self):
        mockContext = initializeMockContext(Mock())
        mockEngine = initializeMockEngine(Mock(), language="en-US")
        handler = EventHandler(Language.EN_US, mockEngine, mockContext)
        for sampleRate in SampleRate:
            requestIterator = asyncStreamingRequestIterator(
                language="en-US",
                sampleRate=sampleRate.value,
                audioEncoding="PCM",
                topic="GENERIC",
                audio=[b"0000"],
            )
            async for request in requestIterator:
                await handler.source(request)
            response = handler.sink(await handler.handle())
            self.assertEqual(
                response.alternatives[0].transcript, DEFAULT_ENGLISH_MESSAGE
            )

    async def testEnUsRecognition(self):
        mockContext = initializeMockContext(Mock())
        mockEngine = initializeMockEngine(Mock(), language="en-US")
        handler = EventHandler(Language.EN_US, mockEngine, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        async for request in requestIterator:
            await handler.source(request)
        response = handler.sink(await handler.handle())
        self.assertEqual(response.alternatives[0].transcript, DEFAULT_ENGLISH_MESSAGE)

    async def testEsRecognition(self):
        mockContext = initializeMockContext(Mock())
        mockEngine = initializeMockEngine(Mock(), language="es")
        handler = EventHandler(Language.ES, mockEngine, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="es",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        async for request in requestIterator:
            await handler.source(request)
        response = handler.sink(await handler.handle())
        self.assertEqual(response.alternatives[0].transcript, DEFAULT_SPANISH_MESSAGE)

    async def testPtBrRecognition(self):
        mockContext = initializeMockContext(Mock())
        mockEngine = initializeMockEngine(Mock(), language="pt-BR")
        handler = EventHandler(Language.PT_BR, mockEngine, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="pt-BR",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        async for request in requestIterator:
            await handler.source(request)
        response = handler.sink(await handler.handle())
        self.assertEqual(
            response.alternatives[0].transcript, DEFAULT_PORTUGUESE_MESSAGE
        )

    async def testEmptyEventSink(self):
        handler = EventHandler(Language.EN_US, None, None)
        response = TranscriptionResult(
            transcription="", score=0.0, words=[], duration=0.0
        )
        result = {
            "results": {
                "alternatives": [
                    {
                        "transcript": "",
                        "confidence": 0.0,
                        "words": [],
                    }
                ],
                "duration": {},
                "end_time": {"seconds": 0, "nanos": 0},
                "is_final": True,
            }
        }
        self.assertEqual(
            handler.sink(response),
            StreamingRecognizeResponse(**result),
        )

    async def testEventSink(self):
        handler = EventHandler(Language.EN_US, None, None)
        handler._totalDuration = 3.4
        response = TranscriptionResult(
            transcription="Hello World!",
            score=1.0,
            words=[
                WordTiming(word="Hello", start=1.0, end=1.5, probability=1.0),
                WordTiming(word="World!", start=1.8, end=2.6, probability=1.0),
            ],
            duration=5,
        )
        result = {
            "results": {
                "alternatives": [
                    {
                        "transcript": "Hello World!",
                        "confidence": 1.0,
                        "words": [
                            {
                                "start_time": {"seconds": 4, "nanos": 400000000},
                                "end_time": {"seconds": 4, "nanos": 900000000},
                                "word": "Hello",
                                "confidence": 1.0,
                            },
                            {
                                "start_time": {"seconds": 5, "nanos": 200000000},
                                "end_time": {"seconds": 6},
                                "word": "World!",
                                "confidence": 1.0,
                            },
                        ],
                    }
                ],
                "duration": {"seconds": 5},
                "end_time": {"seconds": 8, "nanos": 400000000},
                "is_final": True,
            }
        }
        self.assertEqual(
            handler.sink(response),
            StreamingRecognizeResponse(**result),
        )

    async def testRandomEventSink(self):
        handler = EventHandler(Language.EN_US, None, None)
        transcription = " ".join(
            random.choices(string.ascii_letters + string.digits, k=16)
        )
        words = transcription.split()
        response = handler.sink(
            TranscriptionResult(
                transcription=transcription,
                score=1.0,
                words=[
                    WordTiming(
                        word=w, start=float(idx), end=float(idx + 1), probability=1.0
                    )
                    for idx, w in enumerate(words)
                ],
                duration=0.0,
            )
        )
        self.assertEqual(len(response.results.alternatives), 1)
        self.assertEqual(response.results.alternatives[0].transcript, transcription)
        self.assertEqual(response.results.alternatives[0].confidence, 1.0)
        self.assertEqual(len(response.results.alternatives[0].words), 16)
        for idx, w in enumerate(response.results.alternatives[0].words):
            self.assertEqual(w.word, words[idx])
            self.assertEqual(w.start_time.seconds, idx)
            self.assertEqual(w.end_time.seconds, idx + 1)
