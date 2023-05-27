import unittest
from bin import client

class TestRecognizerService(unittest.TestCase):

	def testAudioChunking(self):
		audio_bytes = [i for i in range(10)]
		chunk_iterator = client._chunk_audio(audio_bytes, 3)
		self.assertEqual(list(chunk_iterator), [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])

	def testAudioChunking0(self):
		audio_bytes = [i for i in range(10)]
		chunk_iterator = client._chunk_audio(audio_bytes, 0)
		self.assertEqual(list(chunk_iterator), [audio_bytes])

	def testAudioChunkingEmpty(self):
		audio_bytes = []
		with self.assertRaises(ValueError):
			chunk_iterator = client._chunk_audio(audio_bytes, 3)
			list(chunk_iterator)

	def testAudioChunking0EmptyAudio(self):
		audio_bytes = []
		with self.assertRaises(ValueError):
			chunk_iterator = client._chunk_audio(audio_bytes, 0)
			list(chunk_iterator)