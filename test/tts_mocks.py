

class TTSStubMockCall:
    def code(self):
        return "OK"


class TTSStubMockResponse:
    def __init__(self):
        self.audio_samples = b'your audio samples here!'
