import os
import toml
import unittest
import tempfile
from typing import AsyncIterator
from unittest.mock import patch, Mock

from asr4_streaming.recognizer import RecognizerService
from asr4_streaming.recognizer_v1.types import SampleRate
from asr4_streaming.recognizer import StreamingRecognizeRequest

from tests.unit.test_event_handler import (
    initializeMockEngine,
    initializeMockContext,
    asyncStreamingRequestIterator,
    DEFAULT_ENGLISH_MESSAGE,
    DEFAULT_SPANISH_MESSAGE,
    DEFAULT_PORTUGUESE_MESSAGE,
)


def initializeTomlConfig(language: str = "en-US") -> str:
    fp = tempfile.NamedTemporaryFile(delete=False)
    tomlConfiguration = toml.load("asr4_config.toml")
    tomlConfiguration["global"].setdefault("language", language)
    tomlConfiguration = toml.dumps(tomlConfiguration)
    fp.write(tomlConfiguration.encode())
    fp.seek(0)
    fp.close()
    return fp.name


class TestRecognizerService(unittest.IsolatedAsyncioTestCase):
    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testEmptyRequest(self, mock):
        async def requestIterator() -> AsyncIterator[StreamingRecognizeRequest]:
            yield StreamingRecognizeRequest()
            return

        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator(), mockContext).__anext__()
        self.assertEqual(str(context.exception), "Empty request")

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testInvalidAudio(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b""],
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(str(context.exception), "Empty value for audio")

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testInvalidTopic(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic=-1,
            audio=[b"SOMETHING"],
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(
            str(context.exception), "Invalid value '-1' for topic resource"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testInvalidAudioEncoding(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding=2,
            topic="GENERIC",
            audio=[b"SOMETHING"],
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(
            str(context.exception), "Invalid value '2' for audio_encoding parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testInvalidSampleRate(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        requestIterator = asyncStreamingRequestIterator(
            language="en-US", sampleRate=16001, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(
            str(context.exception), "Invalid value '16001' for sample_rate_hz parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testInvalidLanguage(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        requestIterator = asyncStreamingRequestIterator(
            language="", sampleRate=16000, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(
            str(context.exception), "Invalid value '' for language parameter"
        )

        requestIterator = asyncStreamingRequestIterator(
            language="INVALID", sampleRate=16000, topic="GENERIC", audio=[b"SOMETHING"]
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(
            str(context.exception), "Invalid value 'INVALID' for language parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingConfig(self, mock):
        async def requestIterator() -> AsyncIterator[StreamingRecognizeRequest]:
            yield StreamingRecognizeRequest(audio=b"SOMETHING")
            return

        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator(), mockContext).__anext__()
        self.assertEqual(str(context.exception), "RecognitionConfig was never received")

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingConfigParameters(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        requestIterator = asyncStreamingRequestIterator(topic="GENERIC")
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingConfigParametersExceptLanguage(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        requestIterator = asyncStreamingRequestIterator(language="en-US")
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingConfigParametersExceptEncoding(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        requestIterator = asyncStreamingRequestIterator(audioEncoding="PCM")
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(
            str(context.exception), "Invalid value '0' for sample_rate_hz parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingConfigParametersExceptSampleRate(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        requestIterator = asyncStreamingRequestIterator(sampleRate=8000)
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(
            str(context.exception), "Invalid value '' for language parameter"
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testMissingAudio(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        service = RecognizerService("asr4_config.toml")
        requestIterator = asyncStreamingRequestIterator(
            language="en-US", sampleRate=16000
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(str(context.exception), "Audio was never received")

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testIncorrectLanguage(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        mockContext = initializeMockContext(Mock())
        tomlPath = initializeTomlConfig(language="es")
        service = RecognizerService(tomlPath)
        os.unlink(tomlPath)
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        with self.assertRaises(Exception) as context:
            await service.StreamingRecognize(requestIterator, mockContext).__anext__()
        self.assertEqual(
            str(context.exception), "Invalid language 'en-US'. Only 'es' is supported."
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testRecognitionWithAllSampleRates(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        service = RecognizerService("asr4_config.toml")
        for sampleRate in SampleRate:
            requestIterator = asyncStreamingRequestIterator(
                language="en-US",
                sampleRate=sampleRate.value,
                audioEncoding="PCM",
                topic="GENERIC",
                audio=[b"0000"],
            )
            response = await service.StreamingRecognize(
                requestIterator, None
            ).__anext__()
            self.assertEqual(
                response.results.alternatives[0].transcript, DEFAULT_ENGLISH_MESSAGE
            )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testEnUsRecognition(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="en-US")
        service = RecognizerService("asr4_config.toml")
        requestIterator = asyncStreamingRequestIterator(
            language="en-US",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        response = await service.StreamingRecognize(requestIterator, None).__anext__()
        self.assertEqual(
            response.results.alternatives[0].transcript, DEFAULT_ENGLISH_MESSAGE
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testEsRecognition(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="es")
        tomlPath = initializeTomlConfig(language="es")
        service = RecognizerService(tomlPath)
        os.unlink(tomlPath)
        requestIterator = asyncStreamingRequestIterator(
            language="es",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        response = await service.StreamingRecognize(requestIterator, None).__anext__()
        self.assertEqual(
            response.results.alternatives[0].transcript, DEFAULT_SPANISH_MESSAGE
        )

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    async def testPtBrRecognition(self, mock):
        mock.return_value = initializeMockEngine(Mock(), language="pt-BR")
        tomlPath = initializeTomlConfig(language="pt-BR")
        service = RecognizerService(tomlPath)
        os.unlink(tomlPath)
        requestIterator = asyncStreamingRequestIterator(
            language="pt-BR",
            sampleRate=16000,
            audioEncoding="PCM",
            topic="GENERIC",
            audio=[b"0000"],
        )
        response = await service.StreamingRecognize(requestIterator, None).__anext__()
        self.assertEqual(
            response.results.alternatives[0].transcript, DEFAULT_PORTUGUESE_MESSAGE
        )
