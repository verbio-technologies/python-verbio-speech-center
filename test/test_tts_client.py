from tts_mocks import TTSStubMockResponse, TTSStubMockCall
from helpers.tts_client import TTSClient
from helpers.common import SynthesizerOptions
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
