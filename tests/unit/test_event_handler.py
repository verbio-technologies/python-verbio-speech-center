import grpc
import random
import string
import asyncio
import unittest

from dataclasses import dataclass
from unittest.mock import Mock, AsyncMock
from grpc.aio import ServicerContext, Metadata
from grpc.beta._metadata import _metadatum, _Metadatum
from typing import List, Optional, Union, Iterator, AsyncIterator, Tuple

from google.rpc import code_pb2
from google.rpc.status_pb2 import Status

from asr4_streaming.recognizer import RecognitionConfig
from asr4_streaming.recognizer import RecognitionParameters
from asr4_streaming.recognizer import RecognitionResource
from asr4_streaming.recognizer import StreamingRecognizeRequest
from asr4_streaming.recognizer import StreamingRecognizeResponse

from asr4_streaming.recognizer_v1.types import SampleRate
from asr4_streaming.recognizer_v1.handler import EventHandler
from asr4_streaming.recognizer_v1.handler import TranscriptionResult

from asr4_engine.exceptions import ASR4EngineException
from asr4_engine.data_classes import Language, Transcription
from asr4_engine.data_classes.transcription import WordTiming
from asr4.engines.wav2vec.wav2vec_engine import (
    Wav2VecEngine,
    Wav2VecASR4EngineOnlineHandler,
)

DEFAULT_ENGLISH_MESSAGE: str = "hello i am up and running received a message from you"
DEFAULT_SPANISH_MESSAGE: str = (
    "hola estoy levantado y en marcha y he recibido un mensaje tuyo"
)
FORMATTED_SPANISH_MESSAGE: str = (
    "Hola. Estoy levantado y en marcha y he recibido un mensaje tuyo."
)
DEFAULT_PORTUGUESE_MESSAGE: str = "ola estou de pe recebi uma mensagem sua"


def initializeErrorMockEngine(mock: Mock):
    onlineHandlerMock = AsyncMock(Wav2VecASR4EngineOnlineHandler)

    async def mockListenForCompleteAudio():
        raise ASR4EngineException("Internal error while retrieving timestamps")
        yield

    onlineHandlerMock.listenForCompleteAudio = mockListenForCompleteAudio
    mock.getRecognizerHandler.return_value = onlineHandlerMock
    return mock


def initializeMockEngine(mock: Mock, language: str):
    onlineHandlerMock = AsyncMock(Wav2VecASR4EngineOnlineHandler)

    async def mockListenForCompleteAudio():
        if onlineHandlerMock.sendAudioChunk.called:
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
            t.duration = 10.0
            yield t
        return

    onlineHandlerMock.listenForCompleteAudio = mockListenForCompleteAudio
    mock.getRecognizerHandler.return_value = onlineHandlerMock
    return mock


def initializeMockContext(
    mock: Mock,
    metadata: Metadata = Metadata(
        ("user-id", "testUser"), ("request-id", "testRequest")
    ),
):
    async def abort(_statusCode, message):
        raise grpc.aio.AbortError(message)

    def invocation_metadata() -> Tuple[_Metadatum]:
        return (_metadatum(k, v) for k, v in metadata)

    mock.abort = abort
    mock.invocation_metadata = invocation_metadata
    return mock


async def asyncStreamingRequestIterator(
    language: Optional[str] = None,
    sampleRate: Optional[int] = None,
    audioEncoding: Optional[Union[int, str]] = None,
    topic: Optional[Union[int, str]] = None,
    audio: List[bytes] = [],
    enable_formatting=False,
) -> AsyncIterator[StreamingRecognizeRequest]:
    for message in streamingRequestIterator(
        language, sampleRate, audioEncoding, topic, audio, enable_formatting
    ):
        yield message
    return


def streamingRequestIterator(
    language: Optional[str] = None,
    sampleRate: Optional[int] = None,
    audioEncoding: Optional[Union[int, str]] = None,
    topic: Optional[Union[int, str]] = None,
    audio: List[bytes] = [],
    enable_formatting=False,
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
            enable_formatting=enable_formatting,
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

        mockContext = initializeMockContext(Mock(ServicerContext))
        handler = EventHandler(Language.EN_US, None, mockContext)
        with self.assertRaises(Exception) as context:
            async for request in requestIterator():
                await handler.processStreamingRequest(request)
        self.assertEqual(str(context.exception), "Empty request")

    async def testInvalidAudio(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        mockEngine = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        handler = EventHandler(Language.EN_US, mockEngine, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b""],
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.processStreamingRequest(request)
        self.assertEqual(str(context.exception), "Empty value for audio")

    async def testInvalidTopic(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
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
                await handler.processStreamingRequest(request)
        self.assertEqual(
            str(context.exception), "Invalid value '-1' for topic resource"
        )

    async def testInvalidAudioEncoding(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
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
                await handler.processStreamingRequest(request)
        self.assertEqual(
            str(context.exception), "Invalid value '2' for audio_encoding parameter"
        )

    async def testInvalidSampleRate(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="en-US", sampleRate=16001, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.processStreamingRequest(request)
        self.assertEqual(
            str(context.exception), "Invalid value '16001' for sample_rate_hz parameter"
        )

    async def testInvalidLanguage(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(
            language="", sampleRate=16000, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.processStreamingRequest(request)
        self.assertEqual(
            str(context.exception), "Invalid value '' for language parameter"
        )

        requestIterator = asyncStreamingRequestIterator(
            language="INVALID", sampleRate=16000, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.processStreamingRequest(request)
        self.assertEqual(
            str(context.exception), "Invalid value 'INVALID' for language parameter"
        )

    async def testMissingConfig(self):
        async def requestIterator() -> AsyncIterator[StreamingRecognizeRequest]:
            yield StreamingRecognizeRequest(audio=b"SOMETHING")
            return

        mockContext = initializeMockContext(Mock(ServicerContext))
        handler = EventHandler(Language.EN_US, None, mockContext)
        with self.assertRaises(Exception) as context:
            async for request in requestIterator():
                await handler.processStreamingRequest(request)
        self.assertEqual(
            str(context.exception),
            "A request containing RecognitionConfig must be sent first",
        )

    async def testMissingConfigParameters(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(topic="GENERIC")
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.processStreamingRequest(request)
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    async def testMissingConfigParametersExceptLanguage(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(language="en-US")
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.processStreamingRequest(request)
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    async def testMissingConfigParametersExceptEncoding(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(audioEncoding="PCM")
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.processStreamingRequest(request)
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    async def testMissingConfigParametersExceptSampleRate(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        handler = EventHandler(Language.EN_US, None, mockContext)
        requestIterator = asyncStreamingRequestIterator(sampleRate=8000)
        with self.assertRaises(Exception) as context:
            async for request in requestIterator:
                await handler.processStreamingRequest(request)
        self.assertEqual(
            str(context.exception), "Invalid value '' for language parameter"
        )

    async def testIncorrectLanguage(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
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
                await handler.processStreamingRequest(request)
        self.assertEqual(
            str(context.exception), "Invalid language 'en-US'. Only 'es' is supported."
        )

    async def testMissingAudio(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        mockEngine = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        handler = EventHandler(Language.EN_US, mockEngine, mockContext)
        listenerTask = asyncio.create_task(handler.listenForTranscription())
        requestIterator = asyncStreamingRequestIterator(
            language="en-US", sampleRate=16000
        )
        async for request in requestIterator:
            await handler.processStreamingRequest(request)
        await handler.notifyEndOfAudio()
        await listenerTask
        onlineHandlerMock = mockEngine.getRecognizerHandler()
        onlineHandlerMock.sendAudioChunk.assert_not_called()
        onlineHandlerMock.sendAudioChunk.assert_not_awaited()
        mockContext.write.assert_not_called()
        mockContext.write.assert_not_awaited()

    async def testInternalError(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        mockEngine = initializeErrorMockEngine(Mock(Wav2VecEngine))
        handler = EventHandler(Language.EN_US, mockEngine, mockContext)
        listenerTask = asyncio.create_task(handler.listenForTranscription())
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        async for request in requestIterator:
            await handler.processStreamingRequest(request)
        await handler.notifyEndOfAudio()
        _, _ = await asyncio.wait([listenerTask])

        onlineHandlerMock = mockEngine.getRecognizerHandler()
        onlineHandlerMock.sendAudioChunk.assert_called_once()
        onlineHandlerMock.sendAudioChunk.assert_awaited_once()
        mockContext.write.assert_called_once()
        mockContext.write.assert_awaited_once()
        self.assertEqual(str(listenerTask.exception()), "Internal Server Error")
        response = mockContext.write.call_args.args[0]
        self.assertEqual(
            response,
            StreamingRecognizeResponse(
                error=Status(
                    code=code_pb2.INTERNAL,
                    message="Internal Server Error",
                )
            ),
        )

    async def testRecognitionWithAllSampleRates(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        mockEngine = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        handler = EventHandler(Language.EN_US, mockEngine, mockContext)
        for sampleRate in SampleRate:
            listenerTask = asyncio.create_task(handler.listenForTranscription())
            requestIterator = asyncStreamingRequestIterator(
                language="en-US",
                sampleRate=sampleRate.value,
                audioEncoding="PCM",
                topic="GENERIC",
                audio=[b"0000"],
            )
            async for request in requestIterator:
                await handler.processStreamingRequest(request)
            await handler.notifyEndOfAudio()
            await listenerTask

            onlineHandlerMock = mockEngine.getRecognizerHandler()
            onlineHandlerMock.sendAudioChunk.assert_called_once()
            onlineHandlerMock.sendAudioChunk.assert_awaited_once()
            mockContext.write.assert_called_once()
            mockContext.write.assert_awaited_once()
            response = mockContext.write.call_args.args[0]
            self.assertEqual(
                response.results.alternatives[0].transcript, DEFAULT_ENGLISH_MESSAGE
            )
            mockContext.reset_mock()
            onlineHandlerMock.reset_mock()

    async def testEnUsRecognition(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        mockEngine = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        handler = EventHandler(Language.EN_US, mockEngine, mockContext)
        listenerTask = asyncio.create_task(handler.listenForTranscription())
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        async for request in requestIterator:
            await handler.processStreamingRequest(request)
        await handler.notifyEndOfAudio()
        await listenerTask

        onlineHandlerMock = mockEngine.getRecognizerHandler()
        onlineHandlerMock.sendAudioChunk.assert_called_once()
        onlineHandlerMock.sendAudioChunk.assert_awaited_once()
        mockContext.write.assert_called_once()
        mockContext.write.assert_awaited_once()
        response = mockContext.write.call_args.args[0]
        self.assertEqual(
            response.results.alternatives[0].transcript, DEFAULT_ENGLISH_MESSAGE
        )

    async def testEsRecognition(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        mockEngine = initializeMockEngine(Mock(Wav2VecEngine), language="es")
        handler = EventHandler(Language.ES, mockEngine, mockContext)
        listenerTask = asyncio.create_task(handler.listenForTranscription())
        requestIterator = asyncStreamingRequestIterator(
            language="es",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        async for request in requestIterator:
            await handler.processStreamingRequest(request)
        await handler.notifyEndOfAudio()
        await listenerTask

        onlineHandlerMock = mockEngine.getRecognizerHandler()
        onlineHandlerMock.sendAudioChunk.assert_called_once()
        onlineHandlerMock.sendAudioChunk.assert_awaited_once()
        mockContext.write.assert_called_once()
        mockContext.write.assert_awaited_once()
        response = mockContext.write.call_args.args[0]
        self.assertEqual(
            response.results.alternatives[0].transcript, DEFAULT_SPANISH_MESSAGE
        )

    async def testPtBrRecognition(self):
        mockContext = initializeMockContext(Mock(ServicerContext))
        mockEngine = initializeMockEngine(Mock(Wav2VecEngine), language="pt-BR")
        handler = EventHandler(Language.PT_BR, mockEngine, mockContext)
        listenerTask = asyncio.create_task(handler.listenForTranscription())
        requestIterator = asyncStreamingRequestIterator(
            language="pt-BR",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        async for request in requestIterator:
            await handler.processStreamingRequest(request)
        await handler.notifyEndOfAudio()
        await listenerTask

        onlineHandlerMock = mockEngine.getRecognizerHandler()
        onlineHandlerMock.sendAudioChunk.assert_called_once()
        onlineHandlerMock.sendAudioChunk.assert_awaited_once()
        mockContext.write.assert_called_once()
        mockContext.write.assert_awaited_once()
        response = mockContext.write.call_args.args[0]
        self.assertEqual(
            response.results.alternatives[0].transcript, DEFAULT_PORTUGUESE_MESSAGE
        )

    async def testEmptyEventGetStreamingRecognizeResponse(self):
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
            handler.getStreamingRecognizeResponse(response),
            StreamingRecognizeResponse(**result),
        )

    async def testGetStreamingRecognizeResponse(self):
        handler = EventHandler(Language.EN_US, None, None)
        handler._totalDuration = 3.4
        response = TranscriptionResult(
            transcription="Hello World!",
            score=1.0,
            words=[
                WordTiming(word="Hello", start=1.0, end=1.5, probability=1.0),
                WordTiming(word="World!", start=1.8, end=2.6, probability=1.0),
            ],
            duration=1.6,
        )
        result = {
            "results": {
                "alternatives": [
                    {
                        "transcript": "Hello World!",
                        "confidence": 1.0,
                        "words": [
                            {
                                "start_time": {"seconds": 1},
                                "end_time": {"seconds": 1, "nanos": 500000000},
                                "word": "Hello",
                                "confidence": 1.0,
                            },
                            {
                                "start_time": {"seconds": 1, "nanos": 800000000},
                                "end_time": {"seconds": 2, "nanos": 600000000},
                                "word": "World!",
                                "confidence": 1.0,
                            },
                        ],
                    }
                ],
                "duration": {"seconds": 1, "nanos": 600000000},
                "end_time": {"seconds": 2, "nanos": 600000000},
                "is_final": True,
            }
        }
        self.assertEqual(
            handler.getStreamingRecognizeResponse(response),
            StreamingRecognizeResponse(**result),
        )

    async def testRandomEventGetStreamingRecognizeResponse(self):
        handler = EventHandler(Language.EN_US, None, None)
        transcription = " ".join(
            random.choices(string.ascii_letters + string.digits, k=16)
        )
        words = transcription.split()
        response = handler.getStreamingRecognizeResponse(
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
