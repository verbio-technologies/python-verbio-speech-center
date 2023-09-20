

class TTSStubMockCall:
    def code(self):
        return "OK"


class TTSStubMockResponse:
    def __init__(self):
        self.audio_samples = b'your audio samples here!'


class TTSStubMock:
    class SynthesizeSpeech:
        def with_call(self, request=None, metadata=None):
            response = TTSStubMockResponse()
            call = TTSStubMockCall()
            return (response, call)
