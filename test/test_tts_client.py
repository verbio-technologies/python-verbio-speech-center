from tts_mocks import TTSStubMockResponse, TTSStubMockCall
from helpers.tts_client import TTSClient
from helpers.common import SynthesizerOptions, parse_pronunciation_dict
from unittest.mock import Mock, MagicMock, patch

@patch('helpers.audio_exporter.AudioExporter.save_audio')
def test_synthesis_full_flow_wav(mock_save_audio):
    mock_save_audio.return_value = "mock audio"
    mock_stub = Mock()
    mock_executor = MagicMock()
    mock_stub.SynthesizeSpeech.with_call.return_value = (TTSStubMockResponse(), TTSStubMockCall())
    options = SynthesizerOptions()
    options.audio_format = "wav"
    options.text = "Hello"
    options.sample_rate = 8000
    client = TTSClient(mock_executor, mock_stub, options, "token")
    audio_samples = client.synthesize()
    assert len(audio_samples) == 24


@patch('helpers.audio_exporter.AudioExporter.save_audio')
def test_synthesis_full_flow_raw(mock_save_audio):
    mock_save_audio.return_value = "mock audio"
    mock_stub = Mock()
    mock_executor = MagicMock()
    mock_stub.SynthesizeSpeech.with_call.return_value = (TTSStubMockResponse(), TTSStubMockCall())
    options = SynthesizerOptions()
    options.audio_format = "raw"
    options.sample_rate = 16000
    client = TTSClient(mock_executor, mock_stub, options, "token")
    audio_samples = client.synthesize()
    assert len(audio_samples) == 24


@patch('helpers.audio_exporter.AudioExporter.save_audio')
def test_synthesis_full_flow_wav_unsecured(mock_save_audio):
    mock_save_audio.return_value = "mock audio"
    mock_stub = Mock()
    mock_executor = MagicMock()
    mock_stub.SynthesizeSpeech.with_call.return_value = (TTSStubMockResponse(), TTSStubMockCall())
    options = SynthesizerOptions()
    options.audio_format = "wav"
    options.sample_rate = 8000
    options.secure_channel = False
    client = TTSClient(mock_executor, mock_stub, options, "token")
    audio_samples = client.synthesize()
    assert len(audio_samples) == 24


@patch('helpers.audio_exporter.AudioExporter.save_audio')
def testSynthesisWithPronunciationDictionary(mock_save_audio):
    mock_save_audio.return_value = "mock audio"
    mock_stub = Mock()
    mock_executor = MagicMock()
    mock_stub.SynthesizeSpeech.with_call.return_value = (TTSStubMockResponse(), TTSStubMockCall())
    options = SynthesizerOptions()
    options.audio_format = "wav"
    options.text = "Hello Claughton"
    options.sample_rate = 8000
    options.pronunciation_dictionary = {"Claughton": "ˈklɒftən"}
    client = TTSClient(mock_executor, mock_stub, options, "token")
    audio_samples = client.synthesize()
    assert len(audio_samples) == 24
    request = mock_stub.SynthesizeSpeech.with_call.call_args[0][0]
    assert len(request.pronunciation_dictionary) == 1
    assert request.pronunciation_dictionary[0].term == "Claughton"
    assert request.pronunciation_dictionary[0].ipa == "ˈklɒftən"


@patch('helpers.audio_exporter.AudioExporter.save_audio')
def testSynthesisWithEmptyPronunciationDictionary(mock_save_audio):
    mock_save_audio.return_value = "mock audio"
    mock_stub = Mock()
    mock_executor = MagicMock()
    mock_stub.SynthesizeSpeech.with_call.return_value = (TTSStubMockResponse(), TTSStubMockCall())
    options = SynthesizerOptions()
    options.audio_format = "wav"
    options.text = "Hello"
    options.sample_rate = 8000
    options.pronunciation_dictionary = {}
    client = TTSClient(mock_executor, mock_stub, options, "token")
    audio_samples = client.synthesize()
    assert len(audio_samples) == 24
    request = mock_stub.SynthesizeSpeech.with_call.call_args[0][0]
    assert len(request.pronunciation_dictionary) == 0


@patch('helpers.audio_exporter.AudioExporter.save_audio')
def testSynthesisWithMultiplePronunciationEntries(mock_save_audio):
    mock_save_audio.return_value = "mock audio"
    mock_stub = Mock()
    mock_executor = MagicMock()
    mock_stub.SynthesizeSpeech.with_call.return_value = (TTSStubMockResponse(), TTSStubMockCall())
    options = SynthesizerOptions()
    options.audio_format = "wav"
    options.text = "Hello Claughton, welcome to Karandish"
    options.sample_rate = 16000
    options.pronunciation_dictionary = {"Claughton": "ˈklɒftən", "Karandish": "kəˈrɒndɪʃ"}
    client = TTSClient(mock_executor, mock_stub, options, "token")
    audio_samples = client.synthesize()
    assert len(audio_samples) == 24
    request = mock_stub.SynthesizeSpeech.with_call.call_args[0][0]
    assert len(request.pronunciation_dictionary) == 2
    terms = {e.term: e.ipa for e in request.pronunciation_dictionary}
    assert terms["Claughton"] == "ˈklɒftən"
    assert terms["Karandish"] == "kəˈrɒndɪʃ"


def testParsePronunciationDictFromJsonString():
    raw = '{"Claughton": "ˈklɒftən", "live": "laɪv"}'
    result = parse_pronunciation_dict(raw)
    assert result == {"Claughton": "ˈklɒftən", "live": "laɪv"}


def testParsePronunciationDictReturnsEmptyOnNone():
    assert parse_pronunciation_dict(None) == {}


def testBuildPronunciationEntriesStatic():
    entries = TTSClient._build_pronunciation_entries({"word": "wɜːrd"})
    assert len(entries) == 1
    assert entries[0].term == "word"
    assert entries[0].ipa == "wɜːrd"


def testBuildPronunciationEntriesEmpty():
    entries = TTSClient._build_pronunciation_entries({})
    assert len(entries) == 0
