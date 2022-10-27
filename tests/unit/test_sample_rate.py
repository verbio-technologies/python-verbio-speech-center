import unittest
from asr4.recognizer_v1.types.sample_rate import SampleRate


class TestSampleRate(unittest.TestCase):
    def testParse(self):
        self.assertEqual(SampleRate.parse(16000), SampleRate.HZ_16000)
        self.assertEqual(SampleRate.parse(16001), None)
        self.assertEqual(SampleRate.parse(8000), SampleRate.HZ_8000)
        self.assertEqual(SampleRate.parse(-1), None)
        self.assertEqual(SampleRate.parse(0), None)

    def testCheck(self):
        self.assertTrue(SampleRate.check(16000))
        self.assertFalse(SampleRate.check(16001))
        self.assertTrue(SampleRate.check(8000))
        self.assertFalse(SampleRate.check(-1))
        self.assertFalse(SampleRate.check(0))
