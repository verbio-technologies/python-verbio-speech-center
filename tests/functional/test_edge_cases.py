import os
import grpc
from functools import partial
from typing import Iterator

from google.rpc import code_pb2

from asr4_streaming.recognizer import RecognizerStub
from asr4_streaming.recognizer import StreamingRecognizeRequest

from tests.unit.test_event_handler import streamingRequestIterator
from .recognizer_service_test_case import RecognizerServiceTestCase


class TestEdgeCases(RecognizerServiceTestCase):
    def testEmptyRequest(self):
        def requestIterator() -> Iterator[StreamingRecognizeRequest]:
            yield StreamingRecognizeRequest()
            return

        self._waitForServer()
        channel = grpc.insecure_channel(TestEdgeCases._serverAddress)
        responseIterator = RecognizerStub(channel).StreamingRecognize(
            requestIterator(),
            metadata=(
                ("user-id", "testUser"),
                ("request-id", "testRequest"),
            ),
        )
        response = next(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.INVALID_ARGUMENT)
        self.expectDetails(responseIterator, "Empty request")
        self.assertTrue(response.HasField("error"))
        self.assertEqual(
            response.error.code,
            code_pb2.Code.Value(grpc.StatusCode.INVALID_ARGUMENT.name),
        )
        self.assertEqual(response.error.message, "Empty request")
        with self.assertRaises(grpc.RpcError):
            next(responseIterator)

    def testRecognitionWithMultipleSampleRates(self):
        for audioFile in [
            "0e4b2dbd-95c4-4070-ae6d-e79236e73afb_cut_1-channel-16k.wav",
            "0e4b2dbd-95c4-4070-ae6d-e79236e73afb_cut_1-channel.wav",
        ]:
            responseIterator = self.request(audioFile, "en-US")
            response = RecognizerServiceTestCase.mergeAllResponsesIntoOne(
                responseIterator
            )
            self.expectStatus(responseIterator, grpc.StatusCode.OK)
            expectedResponse = (
                "hi thank you so much for calling international bank this is "
                "becka how can i help you today yeah what up i'm i'm looking to set up a joint "
                "account between me and a child yes we can definitely help you with that over the "
                "phone do you already have an account with us yeah i do perfect so both of these "
                "accounts can definitely be linked can you please give me your full name my phole "
                "name is james barbase"
            )
            self.expectNotEmptyTranscription(response)
            self.expectNumberOfWords(response, len(expectedResponse.split()), 3)
            self.expectTranscriptionWER(
                response, expectedResponse, 0.024691358024691357, delta=4e-2
            )
            self.expectTranscriptionCER(
                response, expectedResponse, 0.004975124378109453, delta=4e-2
            )
            self.expectValidConfidence(response.results.alternatives[0].confidence)
            self.expectDuration(response.results.duration, seconds=1, nanos=960000000)
            self.expectDuration(response.results.end_time, seconds=31, nanos=960000000)
            self.expectFinal(response)
            self.expectValidWords(response)
            self.expectValidWordTimestamps(response, audioDuration=31.96)

    def testRecognitionWithWrongSampleRate(self):
        responseIterator = self.request(
            "0e4b2dbd-95c4-4070-ae6d-e79236e73afb_cut_1-channel.wav",
            "en-US",
            alternativeSampleRate=16000,
        )
        response = RecognizerServiceTestCase.mergeAllResponsesIntoOne(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        self.expectNotEmptyTranscription(response)
        self.expectValidConfidence(response.results.alternatives[0].confidence)
        self.expectDuration(response.results.duration, seconds=5, nanos=980000000)
        self.expectDuration(response.results.end_time, seconds=15, nanos=980000000)
        self.expectFinal(response)
        self.expectValidWords(response)
        self.expectValidWordTimestamps(response, audioDuration=15.98)

    def testRecognitionWithAudioHeader(self):
        def streamingRequestIteratorWithAudioHeader(
            audioFile: str,
        ) -> Iterator[StreamingRecognizeRequest]:
            with open(audioFile, "rb") as f:
                yield from streamingRequestIterator(
                    language="en-US",
                    sampleRate=8000,
                    audio=list(iter(partial(f.read, 1024), b"")),
                )

        requestIterator = streamingRequestIteratorWithAudioHeader(
            os.path.join(
                self.datadir, "0e4b2dbd-95c4-4070-ae6d-e79236e73afb_cut_1-channel.wav"
            ),
        )
        self._waitForServer()
        channel = grpc.insecure_channel(TestEdgeCases._serverAddress)
        responseIterator = RecognizerStub(channel).StreamingRecognize(
            requestIterator,
            metadata=(
                ("user-id", "testUser"),
                ("request-id", "testRequest"),
            ),
        )
        response = RecognizerServiceTestCase.mergeAllResponsesIntoOne(responseIterator)
        self.expectStatus(responseIterator, grpc.StatusCode.OK)
        expectedResponse = (
            "hi thank you so much for calling international bank this is "
            "becka how can i help you today yeah what up i'm i'm looking to set up a joint "
            "account between me and a child yes we can definitely help you with that over the "
            "phone do you already have an account with us yeah i do perfect so both of these "
            "accounts can definitely be linked can you please give me your full name my phole "
            "name is james barbase"
        )
        self.expectNotEmptyTranscription(response)
        self.expectNumberOfWords(response, len(expectedResponse.split()), 5)
        self.expectTranscriptionWER(
            response, expectedResponse, 0.024691358024691357, delta=81e-2
        )
        self.expectTranscriptionCER(
            response, expectedResponse, 0.004975124378109453, delta=64e-2
        )
        self.expectValidConfidence(response.results.alternatives[0].confidence)
        self.expectDuration(response.results.duration, seconds=1, nanos=962750000)
        self.expectDuration(response.results.end_time, seconds=31, nanos=962750000)
        self.expectFinal(response)
        self.expectValidWords(response)
        self.expectValidWordTimestamps(response, audioDuration=31.96)