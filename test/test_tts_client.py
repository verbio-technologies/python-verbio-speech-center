import pytest
from tts_mocks import TTSStubMockResponse, TTSStubMockCall
from helpers.tts_client import TTSClient
from helpers.common import SynthesizerOptions, parse_pronunciation_dict
from unittest.mock import Mock, MagicMock, patch
import speechcenter.tts.text_to_speech_pb2 as text_to_speech_pb2


# --- Helpers ---

def _make_streaming_client(stub, text_file, pronunciation_dictionary=None, voice="marvin_en_us",
                           sample_rate=16000, secure_channel=True):
    executor = MagicMock()
    options = SynthesizerOptions()
    options.text_file = text_file
    options.voice = voice
    options.sample_rate = sample_rate
    options.audio_format = "wav"
    options.audio_file = "/tmp/test_output.wav"
    options.secure_channel = secure_channel
    options.inactivity_timeout = 5.0
    options.pronunciation_dictionary = pronunciation_dictionary or {}
    return TTSClient(executor, stub, options, "token")



# --- Unary synthesis tests ---

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


# --- Streaming synthesis tests ---

def test_streaming_generates_config_text_and_end_of_utterance(tmp_path):
    stub = Mock()
    text_file = tmp_path / "input.txt"
    text_file.write_text("Hello world")
    client = _make_streaming_client(stub, str(text_file))
    client.send_text()

    messages = [msg for _, msg in client._messages]
    assert len(messages) == 3

    assert messages[0].HasField("config")
    assert messages[0].config.voice == "marvin_en_us"
    assert messages[0].config.sampling_rate == text_to_speech_pb2.VoiceSamplingRate.VOICE_SAMPLING_RATE_16KHZ

    assert messages[1].text == "Hello world"

    assert messages[2].HasField("end_of_utterance")
    assert messages[2].end_of_utterance.data == "EndOfUtterance"


def test_streaming_multiple_text_lines(tmp_path):
    stub = Mock()
    text_file = tmp_path / "input.txt"
    text_file.write_text("Line one\nLine two\nLine three")
    client = _make_streaming_client(stub, str(text_file))
    client.send_text()

    messages = [msg for _, msg in client._messages]
    assert len(messages) == 5
    assert messages[1].text == "Line one"
    assert messages[2].text == "Line two"
    assert messages[3].text == "Line three"


def test_streaming_with_pronunciation_dictionary(tmp_path):
    stub = Mock()
    text_file = tmp_path / "input.txt"
    text_file.write_text("Hello Claughton")
    client = _make_streaming_client(stub, str(text_file), pronunciation_dictionary={"Claughton": "ˈklɒftən"})
    client.send_text()

    config = client._messages[0][1].config
    assert len(config.pronunciation_dictionary) == 1
    assert config.pronunciation_dictionary[0].term == "Claughton"
    assert config.pronunciation_dictionary[0].ipa == "ˈklɒftən"


def test_streaming_with_multiple_pronunciation_entries(tmp_path):
    stub = Mock()
    text_file = tmp_path / "input.txt"
    text_file.write_text("Hello Claughton, welcome to Karandish")
    client = _make_streaming_client(stub, str(text_file),
                                    pronunciation_dictionary={"Claughton": "ˈklɒftən", "Karandish": "kəˈrɒndɪʃ"})
    client.send_text()

    config = client._messages[0][1].config
    assert len(config.pronunciation_dictionary) == 2
    terms = {e.term: e.ipa for e in config.pronunciation_dictionary}
    assert terms["Claughton"] == "ˈklɒftən"
    assert terms["Karandish"] == "kəˈrɒndɪʃ"


def test_streaming_with_empty_pronunciation_dictionary(tmp_path):
    stub = Mock()
    text_file = tmp_path / "input.txt"
    text_file.write_text("Hello")
    client = _make_streaming_client(stub, str(text_file), pronunciation_dictionary={})
    client.send_text()

    config = client._messages[0][1].config
    assert len(config.pronunciation_dictionary) == 0


def test_streaming_sampling_rate_8khz(tmp_path):
    stub = Mock()
    text_file = tmp_path / "input.txt"
    text_file.write_text("Hello")
    client = _make_streaming_client(stub, str(text_file), sample_rate=8000)
    client.send_text()

    config = client._messages[0][1].config
    assert config.sampling_rate == text_to_speech_pb2.VoiceSamplingRate.VOICE_SAMPLING_RATE_8KHZ


def test_streaming_unsecured_channel_passes_metadata(tmp_path):
    stub = Mock()
    text_file = tmp_path / "input.txt"
    text_file.write_text("Hello")
    client = _make_streaming_client(stub, str(text_file), secure_channel=False)
    client.send_text()

    call_args = stub.StreamingSynthesizeSpeech.call_args
    metadata = call_args[1].get("metadata") if call_args[1] else call_args[0][1] if len(call_args[0]) > 1 else None
    assert metadata is not None
    assert ("authorization", "Bearer token") in metadata


def test_streaming_secured_channel_no_metadata(tmp_path):
    stub = Mock()
    text_file = tmp_path / "input.txt"
    text_file.write_text("Hello")
    client = _make_streaming_client(stub, str(text_file), secure_channel=True)
    client.send_text()

    call_args = stub.StreamingSynthesizeSpeech.call_args
    metadata = call_args[1].get("metadata") if call_args[1] else None
    assert metadata is None


# --- Utility tests ---

def test_parse_pronunciation_dict_from_json_string():
    raw = '{"Claughton": "ˈklɒftən", "live": "laɪv"}'
    result = parse_pronunciation_dict(raw)
    assert result == {"Claughton": "ˈklɒftən", "live": "laɪv"}


def test_parse_pronunciation_dict_returns_empty_on_none():
    assert parse_pronunciation_dict(None) == {}


def test_parse_pronunciation_dict_returns_empty_on_empty_string():
    assert parse_pronunciation_dict("") == {}


def test_parse_pronunciation_dict_returns_empty_on_whitespace():
    assert parse_pronunciation_dict("   ") == {}


def test_parse_pronunciation_dict_from_file(tmp_path):
    f = tmp_path / "dict.json"
    f.write_text('{"Claughton": "ˈklɒftən"}', encoding="utf-8")
    assert parse_pronunciation_dict(str(f)) == {"Claughton": "ˈklɒftən"}


def test_parse_pronunciation_dict_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_pronunciation_dict("/nonexistent/path/dict.json")


def test_parse_pronunciation_dict_invalid_json_inline():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_pronunciation_dict('{"word": invalid}')


def test_parse_pronunciation_dict_invalid_json_in_file(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text('{"word": invalid}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_pronunciation_dict(str(f))


def test_build_pronunciation_entries_static():
    entries = TTSClient._build_pronunciation_entries({"word": "wɜːrd"})
    assert len(entries) == 1
    assert entries[0].term == "word"
    assert entries[0].ipa == "wɜːrd"


def test_build_pronunciation_entries_empty():
    entries = TTSClient._build_pronunciation_entries({})
    assert len(entries) == 0
