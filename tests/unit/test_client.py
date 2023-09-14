import os
import pytest
import unittest
import argparse

from bin.client import StreamingClient
from asr4_streaming.recognizer import StreamingRecognizeRequest


class TestStreamingClient(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def rootpath(self, pytestconfig):
        self.rootpath = pytestconfig.rootpath

    @pytest.fixture(autouse=True)
    def datapath(self, pytestconfig):
        self.datapath = pytestconfig.rootpath.joinpath("tests/unit/data")

    def setUp(self) -> None:
        self._client = StreamingClient(
            argparse.Namespace(
                gui=None,
                jobs=0,
                host=None,
                batch=False,
                audio=None,
                format=None,
                language=None,
            )
        )

    def tearDown(self) -> None:
        del self._client

    def testAudioChunking(self):
        audioBytes = b"0123456789"
        chunkIterator = self._client._StreamingClient__getAudioStreamingRequests(
            audioBytes, 3
        )
        self.assertEqual(
            list(chunkIterator),
            [
                StreamingRecognizeRequest(audio=b"012"),
                StreamingRecognizeRequest(audio=b"345"),
                StreamingRecognizeRequest(audio=b"678"),
                StreamingRecognizeRequest(audio=b"9"),
            ],
        )

    def testAudioChunking0(self):
        audioBytes = b"0123456789"
        chunkIterator = self._client._StreamingClient__getAudioStreamingRequests(
            audioBytes, 0
        )
        self.assertEqual(
            list(chunkIterator), [StreamingRecognizeRequest(audio=audioBytes)]
        )

    def testAudioChunkingEmpty(self):
        audioBytes = b""
        chunkIterator = self._client._StreamingClient__getAudioStreamingRequests(
            audioBytes, 3
        )
        self.assertEqual(list(chunkIterator), [])

    def testAudioChunking0EmptyAudio(self):
        audioBytes = b""
        chunkIterator = self._client._StreamingClient__getAudioStreamingRequests(
            audioBytes, 0
        )
        self.assertEqual(list(chunkIterator), [])

    def testGetAudio(self):
        _audioBytes, rate, width = self._client._StreamingClient__getAudio(
            os.path.join(self.datapath, "en-us.8k.wav")
        )
        self.assertEqual(width, 2)
        self.assertEqual(rate, 8_000)
        _audioBytes, rate, width = self._client._StreamingClient__getAudio(
            os.path.join(self.datapath, "en-us.16k.wav")
        )
        self.assertEqual(width, 2)
        self.assertEqual(rate, 16_000)
        with self.assertRaises(Exception) as context:
            self._client._StreamingClient__getAudio(
                os.path.join(self.datapath, "en-us.24b.wav")
            )
        self.assertTrue(
            "en-us.24b.wav should have 2-byte samples instead of 3-byte samples."
            in str(context.exception)
        )
