from tts_mocks import TTSStubMockResponse, TTSStubMockCall
from helpers.tts_client import TTSClient
from helpers.common import SynthesizerOptions
from unittest.mock import Mock


def test_synthesis_full_flow_wav():
    mock_stub = Mock()
    mock_stub.SynthesizeSpeech.with_call.return_value = (TTSStubMockResponse(), TTSStubMockCall())
    options = SynthesizerOptions()
    options.audio_format = "wav"
    client = TTSClient(mock_stub, options, "token")
    audio_samples = client.synthesize("Hello my friend!", "tommy_en_us")
    assert len(audio_samples) == 24


def test_synthesis_full_flow_raw():
    mock_stub = Mock()
    mock_stub.SynthesizeSpeech.with_call.return_value = (TTSStubMockResponse(), TTSStubMockCall())
    options = SynthesizerOptions()
    options.audio_format = "raw"
    client = TTSClient(mock_stub, options, "token")
    audio_samples = client.synthesize("Hello my friend!", "miguel_es_pe")
    assert len(audio_samples) == 24


def test_synthesis_full_flow_wav_unsecured():
    mock_stub = Mock()
    mock_stub.SynthesizeSpeech.with_call.return_value = (TTSStubMockResponse(), TTSStubMockCall())
    options = SynthesizerOptions()
    options.audio_format = "wav"
    options.secure_channel = False
    client = TTSClient(mock_stub, options, "token")
    audio_samples = client.synthesize("Olá, meu amigo", "bel_pt_br")
    assert len(audio_samples) == 24

