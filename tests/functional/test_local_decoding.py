import grpc
from .recognizer_service_test_case import RecognizerServiceTestCase


class TestMultipleEnglishLanguageTagsWithLocalDecoding(RecognizerServiceTestCase):
    _kwargs = {
        "decoding_type": "LOCAL",
        "local_formatting": True,
        "formatter": "/mnt/shared/squad2/projects/asr4models/formatter/format-model.en-us-1.1.3.fm",
    }

    def testEnglish(self):
        responseIterator = self.request("en-us-1.wav", "en-US")
        response = RecognizerServiceTestCase.mergeAllResponsesIntoOne(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectNotEmptyTranscription(response)
        self.expectCapitalization(response)
        self.expectNumberOfWords(response, 12, 4)


class TestMultipleSpanishLanguageTagsWithLocalDecoding(RecognizerServiceTestCase):
    _language = "es"
    _kwargs = {
        "decoding_type": "LOCAL",
        "local_formatting": True,
        "formatter": "/mnt/shared/squad2/projects/asr4models/formatter/format-model.es-es-1.1.0.fm",
    }

    def testSpanish(self):
        responseIterator = self.request("es-1.wav", "es")
        response = RecognizerServiceTestCase.mergeAllResponsesIntoOne(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectNotEmptyTranscription(response)
        self.expectCapitalization(response)
        self.expectNumberOfWords(response, 3, 2)


class TestMultiplePortugueseLanguageTagsWithLocalDecoding(RecognizerServiceTestCase):
    _language = "pt-br"
    _kwargs = {
        "decoding_type": "LOCAL",
        "local_formatting": True,
        "formatter": "/mnt/shared/squad2/projects/asr4models/formatter/format-model.pt-br-1.1.1.fm",
    }

    def testPortuguese(self):
        responseIterator = self.request("pt-br-1.wav", "pt-BR")
        response = RecognizerServiceTestCase.mergeAllResponsesIntoOne(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectNotEmptyTranscription(response)
        self.expectCapitalization(response)
        self.expectNumberOfWords(response, 60, 59)
