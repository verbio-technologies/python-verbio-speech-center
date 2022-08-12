import unittest
import random
import string

from asr4.recognizer import RecognizerService
from asr4.recognizer import RecognizeRequest
from asr4.recognizer import RecognitionConfig
from asr4.recognizer import RecognitionParameters
from asr4.recognizer import RecognitionResource
from asr4.recognizer import RecognizeResponse


class TestRecognizerService(unittest.TestCase):
    def testInvalidAudio(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US", sample_rate_hz=16000),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"",
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)


    def testInvalidTopic(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US", sample_rate_hz=16000),
                resource=RecognitionResource(topic=-1),
            ),
            audio=b"SOMETHING",
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)


    def testInvalidLanguage(self):
        service = RecognizerService()
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
                parameters=RecognitionParameters(language="INVALID", sample_rate_hz=16000),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"SOMETHING",
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    
    def testInvalidSampleRate(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US", sample_rate_hz=16001),
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
        service = RecognizerService()
        request = RecognizeRequest()
        with self.assertRaises(ValueError):
            service.eventSource(request)
        
    
    def testInvalidRecognizeRequestAudio(self):
        service = RecognizerService()
        request = RecognizeRequest(audio=b"SOMETHING")
        with self.assertRaises(ValueError):
            service.eventSource(request)

    
    def testInvalidRecognizeRequestResource(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                resource=RecognitionResource(topic="GENERIC")
            )
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    
    def testInvalidRecognizeRequestLanguage(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US"),
            )
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    
    def testInvalidRecognizeRequestSampleRate(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(sample_rate_hz=16000),
            )
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    
    def testInvalidRecognizeRequestParameters(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US", sample_rate_hz=16000),
            )
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)

    
    def testInvalidRecognizeRequestConfig(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US", sample_rate_hz=16000),
                resource=RecognitionResource(topic="GENERIC"),
            )
        )
        with self.assertRaises(ValueError):
            service.eventSource(request)


    def testRecognizeRequest(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US", sample_rate_hz=16000),
                resource=RecognitionResource(topic="GENERIC"),
            ),
            audio=b"SOMETHING",
        )
        self.assertFalse(service.eventSource(request))


    def testInvalidRecognizeRequestHandle(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(),
            )
        )
        self.assertFalse(service.eventHandle(request))

    
    def testRecognizeRequestHandleEnUs(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="en-US"),
            )
        )
        self.assertEqual(service.eventHandle(request), "Hello, I am up and running. Received a message from you!")


    def testRecognizeRequestHandleEsEs(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="es-ES"),
            )
        )
        self.assertEqual(service.eventHandle(request), "Hola, estoy levantado y en marcha. ¡He recibido un mensaje tuyo!")

    
    def testRecognizeRequestHandlePtBr(self):
        service = RecognizerService()
        request = RecognizeRequest(
            config=RecognitionConfig(
                parameters=RecognitionParameters(language="pt-BR"),
            )
        )
        self.assertEqual(service.eventHandle(request), "Olá, estou de pé, recebi uma mensagem sua!")

    
    def testRecognizeRequestSink(self):
        service = RecognizerService()
        response = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        self.assertEqual(service.eventSink(response), RecognizeResponse(text=response))