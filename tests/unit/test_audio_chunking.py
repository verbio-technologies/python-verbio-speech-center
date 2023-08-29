import pytest
import os
import shutil
import unittest
from pathlib import Path
from asr4_streaming.recognizer_v1.types.audio_chunking import (
    AudioChunking,
    loadAudio,
    saveAudio,
    processAudioFile,
    processGuiFile,
)
import wave


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
        self.gui = os.path.join(self.datapath, "audios.gui")
        self.empty_audio = os.path.join(self.datapath, "empty.wav")

    def setUp(self):
        self._output = "testOutput"
        os.mkdir(self._output)

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

    def testEmptyAudio(self):
        with self.assertRaises(Exception):
            loadAudio(self.empty_audio)

    def testSegmentAudioChunk10(self):
        a = AudioChunking(chunkLength=10)
        chunks = a.segmentAudio(loadAudio(self.audio_5_sec_path))
        self.assertEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(
                len(chunk), 80000
            )  # Nº samples = duration * sample_rate = 10 sec * 8000 = 80000
        chunks = a.segmentAudio(loadAudio(self.audio_10_sec_path))
        self.assertEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(
                len(chunk), 80000
            )  # Nº samples = duration * sample_rate = 10 sec * 8000 = 80000
        chunks = a.segmentAudio(loadAudio(self.audio_12_sec_path))
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(
                len(chunk), 80000
            )  # Nº samples = duration * sample_rate = 10 sec * 8000 = 80000

    def testSegmentAudioChunk5(self):
        a = AudioChunking(chunkLength=5)

        chunks = a.segmentAudio(loadAudio(self.audio_5_sec_path))
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(
                len(chunk), 40000
            )  # Nº samples = duration * sample_rate = 5 sec * 8000 = 40000

        chunks = a.segmentAudio(loadAudio(self.audio_10_sec_path))
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(
                len(chunk), 40000
            )  # Nº samples = duration * sample_rate = 5 sec * 8000 = 40000

        chunks = a.segmentAudio(loadAudio(self.audio_12_sec_path))
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(
                len(chunk), 40000
            )  # Nº samples = duration * sample_rate = 5 sec * 8000 = 40000

    def testSoxTrimAudio(self):
        audio = loadAudio(self.audio_12_sec_path)

        chunkLength = 5
        a = AudioChunking(chunkLength)
        sampleRate = 8000
        self.assertEqual(len(a.soxTrimAudio(audio, 0, chunkLength)) / sampleRate, 5)
        self.assertEqual(len(a.soxTrimAudio(audio, 2, chunkLength)) / sampleRate, 5)
        self.assertEqual(len(a.soxTrimAudio(audio, 0, 2)) / sampleRate, 5)

        chunkLength = 10
        a = AudioChunking(10)
        sampleRate = 8000
        self.assertEqual(len(a.soxTrimAudio(audio, 0, chunkLength)) / sampleRate, 10)
        self.assertEqual(len(a.soxTrimAudio(audio, 0, 2)) / sampleRate, 10)

    def testSoxPadAudio(self):
        audio = loadAudio(self.audio_5_sec_path)
        chunkLength = 10
        sampleRate = 8000
        a = AudioChunking(chunkLength)
        self.assertEqual(
            len(a.soxPadAudio(audio["data"], sampleRate, audio["duration"])), 80000
        )  # Nº samples = duration * sample_rate = 10 sec * 8000 = 80000
        chunkLength = 15
        a = AudioChunking(chunkLength)
        self.assertEqual(
            len(a.soxPadAudio(audio["data"], sampleRate, audio["duration"])), 120000
        )  # Nº samples = duration * sample_rate = 15 sec * 8000 = 120000

    def testTrimAudios(self):
        audio = loadAudio(self.audio_12_sec_path)

        chunks = AudioChunking(5).trimAudio(audio)
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(
                len(chunk), 40000
            )  # Nº samples = duration * sample_rate = 5 sec * 8000 = 40000

        chunks = AudioChunking(10).trimAudio(audio)
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(
                len(chunk), 80000
            )  # Nº samples = duration * sample_rate = 10 sec * 8000 = 80000

        chunks = AudioChunking(15).trimAudio(audio)
        self.assertEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(
                len(chunk), 120000
            )  # Nº samples = duration * sample_rate = 15 sec * 8000 = 120000

    def testSaveAudios(self):
        audio = loadAudio(self.audio_12_sec_path)
        audioFile = os.path.join(self._output, Path(self.audio_12_sec_path).stem)
        saveAudio(audio["data"], audio["sample_rate"], audioFile)
        with wave.open(audioFile) as f:
            n = f.getnframes()
            rate = f.getframerate()
            self.assertEqual(rate, 8000)
            self.assertEqual(n / float(rate), 12.08)

    def testGuiFile(self):
        self.assertEqual(len(processGuiFile(self.gui, 10, Path(self._output))), 3)

    def testAudioFile(self):
        self.assertEqual(
            len(processAudioFile(self.audio_5_sec_path, 10, Path(self._output))), 1
        )
        self.assertEqual(
            len(processAudioFile(self.audio_5_sec_path, 5, Path(self._output))), 2
        )
        self.assertEqual(
            len(processAudioFile(self.audio_12_sec_path, 5, Path(self._output))), 3
        )

    def tearDown(self):
        try:
            shutil.rmtree(self._output)
        except OSError as e:
            print("Error removing directory: %s - %s." % (e.filename, e.strerror))
