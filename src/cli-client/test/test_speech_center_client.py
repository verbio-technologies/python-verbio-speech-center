import unittest
import wave

from recognizer import Options as RecognizerOptions, Credentials as RecognizerCredentials, Resources
from synthesizer import Options as SynthesizerOptions, Credentials as SynthesizerCredentials, Audio


class TestRecognizerOptions(unittest.TestCase):
    def test_empty(self):

        options = RecognizerOptions()
        self.assertEqual(options.host, "csr.api.speechcenter.verbio.com")

        with self.assertRaises(Exception) as cm:
            options.check()
        with self.assertRaises(Exception) as cm:
            options.grammar_file = "file"
            options.topic = "generic"
            options.check()
        options.grammar_file = None
        options.topic = "generic"
        options.check()
        options.topic = None
        options.grammar_file = "file"
        options.check()


class TestSynthesizerOptions(unittest.TestCase):
    def test_empty(self):
        options = SynthesizerOptions()
        self.assertEqual(options.host, "tts.api.speechcenter.verbio.com")

        with self.assertRaises(Exception) as cm:
            options.check()
        with self.assertRaises(Exception) as cm:
            options.encoding = "invalid"
            options.check()
        with self.assertRaises(Exception) as cm:
            options.encoding = "PCM"
            options.header = "invalid"
            options.check()
        options.header = "wav"
        options.voice = "Tommy"
        options.check()

    def test_voice_en_us(self):
        options = SynthesizerOptions()
        with self.assertRaises(Exception) as cm:
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Aurora"
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "David"
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Luma"
            options.check()
        options.voice = "Tommy"
        options.check()
        options.voice = "Annie"
        options.check()

    def test_voice_es_es(self):
        options = SynthesizerOptions()
        options.language = 'es-ES'
        with self.assertRaises(Exception) as cm:
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Tommy"
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Annie"
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Luma"
            options.check()
        options.voice = "Aurora"
        options.check()
        options.voice = "David"
        options.check()

    def test_voice_pt_br(self):
        options = SynthesizerOptions()
        options.language = 'pt-BR'
        with self.assertRaises(Exception) as cm:
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Tommy"
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Annie"
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Aurora"
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "David"
            options.check()
        options.voice = "Luma"
        options.check()

    def test_voice_ca_ca(self):
        options = SynthesizerOptions()
        options.language = 'ca-CA'
        with self.assertRaises(Exception) as cm:
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Tommy"
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Annie"
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Aurora"
            options.check()
        with self.assertRaises(Exception) as cm:
            options.voice = "Luma"
            options.check()
        options.voice = "David"
        options.check()


class TestCredentials(unittest.TestCase):
    def test_init(self):
        credentials = RecognizerCredentials("añlsdfjkñasldfkjñasldkf")
        channel_credentials = credentials.get_channel_credentials()
        self.assertIsNotNone(channel_credentials._credentials)
        credentials = SynthesizerCredentials("añlsdfjkñasldfkjñasldkf")
        channel_credentials = credentials.get_channel_credentials()
        self.assertIsNotNone(channel_credentials._credentials)


class TestResources(unittest.TestCase):
    wav_content = b'data'
    bnf_content = 'content'

    def test_init(self):
        options = RecognizerOptions()
        options.grammar_file = "file.bnf"
        options.audio_file = "file.wav"
        options.check()

    def test_grammar_file_empty(self):
        options = RecognizerOptions()
        options.audio_file = self.__write_file_wav()
        resources = Resources(options)
        self.assertEqual(self.wav_content, resources.audio)

    def test_grammar_file(self):
        options = RecognizerOptions()
        options.grammar_file = self.__write_bnf()
        options.audio_file = self.__write_file_wav()
        resources = Resources(options)
        self.assertEqual(self.bnf_content, resources.grammar)

    def __write_file_wav(self) -> str:
        filename = 'file.wav'
        with open(filename, 'wb') as file_hdl:
            wav_hdl = wave.open(file_hdl, 'w')
            wav_hdl.setnchannels(1)
            wav_hdl.setsampwidth(2)
            wav_hdl.setframerate(8000)
            wav_hdl.writeframes(self.wav_content)
            wav_hdl.close()
        return filename

    def __write_bnf(self) -> str:
        filename = 'file.bnf'
        with open(filename, 'w') as bnf_hdl:
            bnf_hdl.write(self.bnf_content)
        return filename


class TestAudio(unittest.TestCase):
    wav_content = b'data'

    def test_speaker(self):
        options = SynthesizerOptions()
        options.voice = "Tommy"
        self.assertEqual(Audio(options).speaker, 'EN_US_TOMMY')
        options.voice = "Annie"
        self.assertEqual(Audio(options).speaker, 'EN_US_ANNIE')
        options.language = "es-ES"
        options.voice = "Aurora"
        self.assertEqual(Audio(options).speaker, 'ES_ES_AURORA')
        options.voice = "David"
        self.assertEqual(Audio(options).speaker, 'ES_ES_DAVID')
        options.language = "ca-CA"
        self.assertEqual(Audio(options).speaker, 'CA_CA_DAVID')
        options.language = "pt-BR"
        options.voice = "Luma"
        self.assertEqual(Audio(options).speaker, 'PT_BR_LUMA')
    
    def test_sample_rate(self):
        options = SynthesizerOptions()
        options.voice = ''
        self.assertEqual(Audio(options).sample_rate, 0)

    def test_audio_format(self):
        options = SynthesizerOptions()
        options.voice = ''
        self.assertEqual(Audio(options).audio_format, 0)
        options.header = "raw"
        self.assertEqual(Audio(options).audio_format, 1)

    def test_save_audio(self):
        options = SynthesizerOptions()
        options.voice = ''
        audio = Audio(options)
        audio.save_audio(self.wav_content, 'file.wav')
        self.assertEqual(self.wav_content, self.__read_file_wav('file.wav'))
        options.header = "raw"
        audio = Audio(options)
        audio.save_audio(self.wav_content, 'file.raw')
        self.assertEqual(self.wav_content, self.__read_file_raw('file.raw'))

    def __read_file_wav(self, filename: str) -> bytes:
        with open(filename, "rb") as wav_file:
            wav_data = wave.open(wav_file)
            audio_data = wav_data.readframes(wav_data.getnframes())
            return audio_data

    def __read_file_raw(self, filename: str) -> bytes:
        with open(filename, "rb") as f:
            return f.read()

if __name__ == '__main__':
    unittest.main()
