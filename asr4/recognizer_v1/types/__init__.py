from .asr4_pb2 import (
    RecognizeRequest,
    RecognizeResponse,
    RecognitionConfig,
    RecognitionParameters,
    RecognitionResource,
)

from .asr4_pb2_grpc import (
    RecognizerStub,
    RecognizerServicer,
    add_RecognizerServicer_to_server,
)

SERVICES_NAMES = [
    service.full_name for service in asr4_pb2.DESCRIPTOR.services_by_name.values()
]

__all__ = (
    "SERVICES_NAMES",
    "RecognizeRequest",
    "RecognizeResponse",
    "RecognitionConfig",
    "RecognitionParameters",
    "RecognitionResource",
    "RecognizerStub",
    "RecognizerServicer",
    "add_RecognizerServicer_to_server",
)
