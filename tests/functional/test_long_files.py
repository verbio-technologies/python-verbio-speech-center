import grpc
from .recognizer_service_test_case import RecognizerServiceTestCase


class TestEnglishLongFiles(RecognizerServiceTestCase):
    def testEnglish120m(self):
        responseIterator = self.request("historyofengland_120m.wav", "en-US")
        response = self.mergeAllResponsesIntoOne(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectNotEmptyTranscription(response)
        self.expectNumberOfWords(response, 13200, 3000)


class TestSpanishLongFiles(RecognizerServiceTestCase):
    _language = "es"

    def testSpanish127m(self):
        responseIterator = self.request("20000leguas_127m.wav", "es")
        response = self.mergeAllResponsesIntoOne(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectNotEmptyTranscription(response)
        self.expectNumberOfWords(response, 15000, 2000)


class TestPortugueseLongFiles(RecognizerServiceTestCase):
    _language = "pt-br"

    def testPortuguese120m(self):
        responseIterator = self.request("margemdahistoria_120m.wav", "pt-BR")
        response = self.mergeAllResponsesIntoOne(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectNotEmptyTranscription(response)
        self.expectNumberOfWords(response, 13200, 3000)
