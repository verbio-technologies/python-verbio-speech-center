import argparse
import pytest
import os
import unittest
from asr4.recognizer_v1.types.audio_chunking import AudioChunking, loadAudio, saveAudio

from pathlib import Path


class TestAudioChunking(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def rootpath(self, pytestconfig):
        self.rootpath = pytestconfig.rootpath
        self.datapath = pytestconfig.rootpath.joinpath("tests/unit/data")

        self.audio_5_sec_path = os.path.join(self.datapath, "5.07s.wav")
        self.audio_10_sec_path = os.path.join(self.datapath, "10s.wav")
        self.audio_12_sec_path = os.path.join(self.datapath, "12.08s.wav")
        self.audio_8k_path = os.path.join(self.datapath, "en-us.8k.wav")
        self.audio_16k_path = os.path.join(self.datapath, "en-us.16k.wav")

    def testLoadAudioDifferentLength(self):
        self.assertEqual(loadAudio(self.audio_5_sec_path)["duration"], 5.0735)
        self.assertEqual(len(loadAudio(self.audio_5_sec_path)["data"]), 40588)
        self.assertEqual(loadAudio(self.audio_5_sec_path)["sample_rate"], 8000)
        self.assertEqual(loadAudio(self.audio_10_sec_path)["duration"], 10)
        self.assertEqual(len(loadAudio(self.audio_10_sec_path)["data"]), 80000)
        self.assertEqual(loadAudio(self.audio_5_sec_path)["sample_rate"], 8000)
        self.assertEqual(loadAudio(self.audio_12_sec_path)["duration"], 12.08)
        self.assertEqual(len(loadAudio(self.audio_12_sec_path)["data"]), 96640)
        self.assertEqual(loadAudio(self.audio_5_sec_path)["sample_rate"], 8000)

    def testLoadAudioDifferentSampleRate(self):
        self.assertEqual(loadAudio(self.audio_8k_path)["sample_rate"], 8000)
        self.assertEqual(loadAudio(self.audio_16k_path)["sample_rate"], 16000)
        self.assertEqual(
            loadAudio(self.audio_8k_path)["duration"],
            loadAudio(self.audio_16k_path)["duration"],
        )

    def testSegmentAudioChunk10(self):
        a = AudioChunking(chunkLength=10)
        self.assertEqual(len(a.segmentAudio(loadAudio(self.audio_5_sec_path))), 1)
        self.assertEqual(len(a.segmentAudio(loadAudio(self.audio_10_sec_path))), 1)
        self.assertEqual(len(a.segmentAudio(loadAudio(self.audio_12_sec_path))), 2)

    def testSegmentAudioChunk5(self):
        a = AudioChunking(chunkLength=5)
        self.assertEqual(len(a.segmentAudio(loadAudio(self.audio_5_sec_path))), 2)
        self.assertEqual(len(a.segmentAudio(loadAudio(self.audio_10_sec_path))), 2)
        self.assertEqual(len(a.segmentAudio(loadAudio(self.audio_12_sec_path))), 3)
