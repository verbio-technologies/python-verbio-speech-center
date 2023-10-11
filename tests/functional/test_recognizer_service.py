import io
import os
import pytest
import unittest
from loguru import logger
from typing import AsyncIterator
from unittest.mock import patch, Mock
from grpc.aio import ServicerContext, Metadata

from asr4_streaming.recognizer import RecognizerService
from asr4_streaming.recognizer_v1.types import SampleRate
from asr4_streaming.recognizer import StreamingRecognizeRequest
from asr4.engines.wav2vec.wav2vec_engine import Wav2VecEngine

from tests.unit.test_event_handler import (
    initializeMockEngine,
    initializeMockContext,
    initializeErrorMockEngine,
    asyncStreamingRequestIterator,
    DEFAULT_ENGLISH_MESSAGE,
    DEFAULT_SPANISH_MESSAGE,
    DEFAULT_PORTUGUESE_MESSAGE,
)


@pytest.mark.usefixtures("datadir")
class TestRecognizerService(unittest.IsolatedAsyncioTestCase):
    @pytest.fixture(autouse=True)
    def datadir(self, pytestconfig):
        self.datadir = f"{pytestconfig.rootdir}/tests/functional/data"

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testEmptyRequest(self, mock):
        async def requestIterator() -> AsyncIterator[StreamingRecognizeRequest]:
            yield StreamingRecognizeRequest()
            return

        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator(), mockContext)
        self.assertEqual(str(context.exception), "Empty request")

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testNoUserAndRequestId(self, mock):
        async def requestIterator() -> AsyncIterator[StreamingRecognizeRequest]:
            yield StreamingRecognizeRequest()
            return

        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext), metadata=Metadata())
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator(), mockContext)
        self.assertEqual(str(context.exception), "Empty request")

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testInvalidAudio(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b""],
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(str(context.exception), "Empty value for audio")

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testInvalidTopic(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic=-1,
            audio=[b"SOMETHING"],
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(
            str(context.exception), "Invalid value '-1' for topic resource"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testInvalidAudioEncoding(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding=2,
            topic="GENERIC",
            audio=[b"SOMETHING"],
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(
            str(context.exception), "Invalid value '2' for audio_encoding parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testInvalidSampleRate(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(
            language="en-US", sampleRate=16001, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(
            str(context.exception), "Invalid value '16001' for sample_rate_hz parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testInvalidLanguage(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(
            language="", sampleRate=16000, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(
            str(context.exception), "Invalid value '' for language parameter"
        )

        requestIterator = asyncStreamingRequestIterator(
            language="INVALID", sampleRate=16000, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(
            str(context.exception), "Invalid value 'INVALID' for language parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingConfig(self, mock):
        async def requestIterator() -> AsyncIterator[StreamingRecognizeRequest]:
            yield StreamingRecognizeRequest(audio=b"SOMETHING")
            return

        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator(), mockContext)
        self.assertEqual(
            str(context.exception),
            "A request containing RecognitionConfig must be sent first",
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingConfigParameters(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(topic="GENERIC")
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingConfigParametersExceptLanguage(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(language="en-US")
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingConfigParametersExceptEncoding(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(audioEncoding="PCM")
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingConfigParametersExceptSampleRate(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(sampleRate=8000)
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(
            str(context.exception), "Invalid value '' for language parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingAudio(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(
            language="en-US", sampleRate=16000
        )

        await service.StreamingRecognize(requestIterator, mockContext)
        onlineHandlerMock = mock().getRecognizerHandler()
        onlineHandlerMock.sendAudioChunk.assert_not_called()
        onlineHandlerMock.sendAudioChunk.assert_not_awaited()
        mockContext.write.assert_not_called()
        mockContext.write.assert_not_awaited()

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testIncorrectLanguage(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_es.toml")
        )
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(
            str(context.exception), "Invalid language 'en-US'. Only 'es' is supported."
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testInternalError(self, mock):
        logBuffer = io.StringIO()
        handlerId = logger.add(logBuffer, level="ERROR")
        mock.return_value = initializeErrorMockEngine(Mock(Wav2VecEngine))
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext)
        self.assertEqual(str(context.exception), "Internal Server Error")
        self.assertIn(
            "Internal error while retrieving timestamps",
            logBuffer.getvalue(),
        )
        logger.remove(handlerId)

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testRecognitionWithAllSampleRates(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        for sampleRate in SampleRate:
            requestIterator = asyncStreamingRequestIterator(
                language="en-US",
                sampleRate=sampleRate.value,
                audioEncoding="PCM",
                topic="GENERIC",
                audio=[b"0000"],
            )

            await service.StreamingRecognize(requestIterator, mockContext)
            onlineHandlerMock = mock().getRecognizerHandler()
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

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testEnUsRecognition(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )

        await service.StreamingRecognize(requestIterator, mockContext)
        onlineHandlerMock = mock().getRecognizerHandler()
        onlineHandlerMock.sendAudioChunk.assert_called_once()
        onlineHandlerMock.sendAudioChunk.assert_awaited_once()
        mockContext.write.assert_called_once()
        mockContext.write.assert_awaited_once()
        response = mockContext.write.call_args.args[0]
        self.assertEqual(
            response.results.alternatives[0].transcript, DEFAULT_ENGLISH_MESSAGE
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testEsRecognition(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="es")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_es.toml")
        )
        requestIterator = asyncStreamingRequestIterator(
            language="es",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )

        await service.StreamingRecognize(requestIterator, mockContext)
        onlineHandlerMock = mock().getRecognizerHandler()
        onlineHandlerMock.sendAudioChunk.assert_called_once()
        onlineHandlerMock.sendAudioChunk.assert_awaited_once()
        mockContext.write.assert_called_once()
        mockContext.write.assert_awaited_once()
        response = mockContext.write.call_args.args[0]
        self.assertEqual(
            response.results.alternatives[0].transcript, DEFAULT_SPANISH_MESSAGE
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testPtBrRecognition(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="pt-BR")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_pt-br.toml")
        )
        requestIterator = asyncStreamingRequestIterator(
            language="pt-BR",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )

        await service.StreamingRecognize(requestIterator, mockContext)
        onlineHandlerMock = mock().getRecognizerHandler()
        onlineHandlerMock.sendAudioChunk.assert_called_once()
        onlineHandlerMock.sendAudioChunk.assert_awaited_once()
        mockContext.write.assert_called_once()
        mockContext.write.assert_awaited_once()
        response = mockContext.write.call_args.args[0]
        self.assertEqual(
            response.results.alternatives[0].transcript, DEFAULT_PORTUGUESE_MESSAGE
        )
