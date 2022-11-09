from .asr4_pb2 import (
    RecognizeRequest,
    StreamingRecognizeRequest,
    RecognizeResponse,
    StreamingRecognizeResponse,
    StreamingRecognitionResult,
    RecognitionAlternative,
    WordInfo,
    RecognitionConfig,
    RecognitionParameters,
    RecognitionResource,
)

from .asr4_pb2_grpc import (
    RecognizerStub,
    RecognizerServicer,
    add_RecognizerServicer_to_server,
)

from .sample_rate import SampleRate
from google.protobuf.duration_pb2 import Duration

SERVICES_NAMES = [
    service.full_name for service in asr4_pb2.DESCRIPTOR.services_by_name.values()
]


__all__ = (
    "SERVICES_NAMES",
    "RecognizeRequest",
    "StreamingRecognizeRequest",
    "RecognizeResponse",
    "StreamingRecognizeResponse",
    "StreamingRecognitionResult",
    "RecognitionAlternative",
    "Duration",
    "WordInfo",
    "RecognitionConfig",
    "RecognitionParameters",
    "RecognitionResource",
    "RecognizerStub",
    "RecognizerServicer",
    "add_RecognizerServicer_to_server",
    "SampleRate",
)
