import grpc
from .recognizer_service_test_case import RecognizerServiceTestCase


class TestMultipleEnglishLanguageTags(RecognizerServiceTestCase):
    def testEnglish(self):
        responseIterator = self.request("en-us-1.wav", "en-US")
        response = next(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectNotEmptyTranscription(response)
        self.expectNumberOfWords(response, 12, 4)


class TestMultipleSpanishLanguageTags(RecognizerServiceTestCase):
    _language = "es"

    def testSpanish(self):
        responseIterator = self.request("es-1.wav", "es")
        response = next(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectNotEmptyTranscription(response)
        self.expectNumberOfWords(response, 3, 2)


class TestMultiplePortugueseLanguageTags(RecognizerServiceTestCase):
    _language = "pt-br"

    def testPortuguese(self):
        responseIterator = self.request("pt-br-1.wav", "pt-BR")
        response = next(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectNotEmptyTranscription(response)
        self.expectNumberOfWords(response, 60, 59)
