from .asr4_pb2 import (
    RecognizeRequest,
    StreamingRecognizeRequest,
    RecognizeResponse,
    StreamingRecognizeResponse,
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

SERVICES_NAMES = [
    service.full_name for service in asr4_pb2.DESCRIPTOR.services_by_name.values()
]


__all__ = (
    "SERVICES_NAMES",
    "RecognizeRequest",
    "StreamingRecognizeRequest",
    "RecognizeResponse",
    "StreamingRecognizeResponse",
    "RecognitionConfig",
    "RecognitionParameters",
    "RecognitionResource",
    "RecognizerStub",
    "RecognizerServicer",
    "add_RecognizerServicer_to_server",
    "SampleRate",
)
