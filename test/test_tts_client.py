from tts_mocks import TTSStubMockResponse, TTSStubMockCall
from helpers.tts_client import TTSClient
from helpers.common import SynthesizerOptions
from unittest.mock import Mock


def test_synthesis_full_flow():
    mock_stub = Mock()
    mock_stub.SynthesizeSpeech.with_call.return_value = (TTSStubMockResponse(), TTSStubMockCall())
    options = SynthesizerOptions()
    options.audio_format = "wav"
    client = TTSClient(mock_stub, options, "token")
    audio_samples = client.synthesize("Hello my friend!", "tommy_en_us")
    assert len(audio_samples) == 24
