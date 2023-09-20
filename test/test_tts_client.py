from tts_mocks import TTSStubMock
from helpers.tts_client import TTSClient
from helpers.common import SynthesizerOptions


def test_synthesis_full_flow():
    options = SynthesizerOptions()
    options.audio_format = "wav"
    mock_stub = TTSStubMock()
    client = TTSClient(mock_stub, options, "token")
    audio_samples = client.synthesize("Hello my friend!", "tommy_en_us")
    assert len(audio_samples) == 24
