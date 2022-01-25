import unittest
import wave

from cli_client import Options, Credentials, Resources


class TestOptions(unittest.TestCase):
    def test_empty(self):

        options = Options()
        self.assertEqual(options.host, "speechcenter.verbio.com:2424")

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


class TestCredentials(unittest.TestCase):
    def test_init(self):
        credentials = Credentials("añlsdfjkñasldfkjñasldkf")
        channel_credentials = credentials.get_channel_credentials()
        self.assertIsNotNone(channel_credentials._credentials)


class TesResources(unittest.TestCase):
    wav_content = b'data'
    bnf_contnet = 'content'

    def test_init(self):
        options = Options()
        options.grammar_file = "file.bnf"
        options.audio_file = "file.wav"

    def test_grammar_file_empty(self):
        options = Options()
        options.audio_file = self.__write_file_wav()
        resources = Resources(options)
        self.assertEqual(self.wav_content, resources.audio)

    def test_grammar_file(self):
        options = Options()
        options.grammar_file = self.__write_bnf()
        options.audio_file = self.__write_file_wav()
        resources = Resources(options)
        self.assertEqual(self.bnf_contnet, resources.grammar)

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
            bnf_hdl.write(self.bnf_contnet)
        return filename


if __name__ == '__main__':
    unittest.main()
