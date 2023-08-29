import unittest
from asr4_streaming.recognizer_v1.types.audio_encoding import AudioEncoding


class TestAudioEncoding(unittest.TestCase):
    def testParse(self):
        self.assertEqual(AudioEncoding.parse(0), AudioEncoding.PCM)
        self.assertEqual(AudioEncoding.parse(1), None)
        self.assertEqual(AudioEncoding.parse(-1), None)
        self.assertEqual(AudioEncoding.parse(2), None)

    def testCheck(self):
        self.assertTrue(AudioEncoding.check(0))
        self.assertFalse(AudioEncoding.check(1))
        self.assertFalse(AudioEncoding.check(-1))
        self.assertFalse(AudioEncoding.check(2))
