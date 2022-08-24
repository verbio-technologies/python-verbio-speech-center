import unittest
import random
import string
import numpy as np

from asr4.recognizer import RecognizerService
from asr4.recognizer import RecognizeRequest
from asr4.recognizer import RecognitionConfig
from asr4.recognizer import RecognitionParameters
from asr4.recognizer import RecognitionResource
from asr4.recognizer import RecognizeResponse
from asr4.recognizer import Session, OnnxRuntime

from typing import Any, Dict, List, Optional, Union

DEFAULT_ENGLISH_MESSAGE: str = "hello i am up and running received a message from you"
DEFAULT_SPANISH_MESSAGE: str = (
    "Hola, estoy levantado y en marcha. ¡He recibido un mensaje tuyo!"
)
DEFAULT_PORTUGUESE_MESSAGE: str = "Olá, estou de pé, recebi uma mensagem sua!"


class MockOnnxSession(Session):
    def __init__(
        self,
        _path_or_bytes: Union[str, bytes],
        **kwargs,
    ) -> None:
        pass

    def run(
        self,
        _output_names: Optional[List[str]],
        input_feed: Dict[str, Any],
        **kwargs,
    ) -> np.ndarray:
        englishMessage = list(DEFAULT_ENGLISH_MESSAGE.replace(" ", "|"))
        return [self._generateDefaultMessageArray(englishMessage)]

    def _generateDefaultMessageArray(self, defaultMessage: List[str]) -> np.ndarray:
        defaultMessageArray = np.full(
            (1, len(defaultMessage), len(OnnxRuntime.DEFAULT_VOCABULARY)),
            -10.0,
            np.float32,
        )
        for (i, letter) in enumerate(defaultMessage):
            defaultMessageArray[
                0, i, OnnxRuntime.DEFAULT_VOCABULARY.index(letter)
            ] = 10.0
        return self._insertBlankBetweenRepeatedLetters(
            defaultMessage, defaultMessageArray
        )

    def _insertBlankBetweenRepeatedLetters(
        self, defaultMessage: List[str], defaultMessageArray: np.ndarray
    ) -> np.ndarray:
        lastLetter, offset = "", 0
        blank_row = self._getBlankArray()
        for (i, letter) in enumerate(defaultMessage):
            if lastLetter == letter:
                defaultMessageArray = np.insert(
                    defaultMessageArray, i + offset, blank_row, axis=1
                )
                offset += 1
            lastLetter = letter
        return defaultMessageArray

    def _getBlankArray(self) -> np.ndarray:
        blank_row = np.zeros(len(OnnxRuntime.DEFAULT_VOCABULARY), dtype=np.float32)
        blank_row[OnnxRuntime.DEFAULT_VOCABULARY.index("<s>")] = 10.0
        return blank_row

    def get_inputs_names(self) -> List[str]:
        return ["input"]


class TestRecognizerService(unittest.TestCase):
    def testInvalidAudio(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(
                    language="en-US", sample_rate_hz=16000
                ),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"",
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    def testInvalidTopic(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(
                    language="en-US", sample_rate_hz=16000
                ),
                resource=RecognitionResource(topic=-1),
            ),
            audio=b"SOMETHING",
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    def testInvalidLanguage(self):
        service = RecognizerService(MockOnnxSession(""))
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
        service = RecognizerService(MockOnnxSession(""))
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
                parameters=RecognitionParameters(language="en-US", sample_rate_hz=8000),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"SOMETHING",
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    def testInvalidRecognizeRequestEmpty(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest()
        with self.assertRaises(ValueError):
            service.eventSource(request)

    def testInvalidRecognizeRequestAudio(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(audio=b"SOMETHING")
        with self.assertRaises(ValueError):
            service.eventSource(request)

    def testInvalidRecognizeRequestResource(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(resource=RecognitionResource(topic="GENERIC"))
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    def testInvalidRecognizeRequestLanguage(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US"),
            )
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    def testInvalidRecognizeRequestSampleRate(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(sample_rate_hz=16000),
            )
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    def testInvalidRecognizeRequestParameters(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(
                    language="en-US", sample_rate_hz=16000
                ),
            )
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    def testInvalidRecognizeRequestConfig(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(
                    language="en-US", sample_rate_hz=16000
                ),
                resource=RecognitionResource(topic="GENERIC"),
            )
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    def testRecognizeRequest(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(
                    language="en-US", sample_rate_hz=16000
                ),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"SOMETHING",
        )
        self.assertFalse(service.eventSource(request))

    def testInvalidRecognizeRequestHandle(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(),
            )
        )
        self.assertFalse(service.eventHandle(request))

    def testRecognizeRequestHandleEnUs(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US"),
            ),
            audio=b"0000",
        )
        self.assertEqual(
            service.eventHandle(request),
            DEFAULT_ENGLISH_MESSAGE,
        )

    def testRecognizeRequestHandleEsEs(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="es-ES"),
            )
        )
        self.assertEqual(
            service.eventHandle(request),
            DEFAULT_SPANISH_MESSAGE,
        )

    def testRecognizeRequestHandlePtBr(self):
        service = RecognizerService(MockOnnxSession(""))
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="pt-BR"),
            )
        )
        self.assertEqual(service.eventHandle(request), DEFAULT_PORTUGUESE_MESSAGE)

    def testRecognizeRequestSink(self):
        service = RecognizerService(MockOnnxSession(""))
        response = "".join(random.choices(string.ascii_letters + string.digits, k=16))
        self.assertEqual(service.eventSink(response), RecognizeResponse(text=response))
