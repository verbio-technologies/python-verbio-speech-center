import grpc
from .recognizer_service_test_case import RecognizerServiceTestCase


class TestStereoFile(RecognizerServiceTestCase):
    def testStereoFileTwoSpeakers(self):
        responseIterator = self.request("stereo-en_US.wav", "en-US")
        response = next(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectNotEmptyTranscription(response)
        # the audio has 20 words in one channel and 30 words in the other channel aprox.
        self.expectNumberOfWords(response, 1, 10)

    def testRecognitionWithStereoAudio(self):
        responseIterator = self.request(
            "0e4b2dbd-95c4-4070-ae6d-e79236e73afb_cut.wav", "en-US"
        )
        response = next(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectValidConfidence(response.results.alternatives[0].confidence)
        self.expectDuration(response.results.duration, seconds=63, nanos=920000000)
        self.expectDuration(response.results.end_time, seconds=63, nanos=920000000)
        self.expectFinal(response)
        self.expectValidWords(response)
