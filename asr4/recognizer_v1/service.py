import abc
import grpc
import logging

from .types import RecognizerServicer
from .types import RecognizeRequest
from .types import RecognitionConfig
from .types import RecognitionParameters
from .types import RecognitionResource
from .types import RecognizeResponse
from .types import Language
from .types import SampleRate

from google.protobuf.reflection import GeneratedProtocolMessageType


class SourceSinkService(abc.ABC):
    def eventSource(
        self,
        _request: GeneratedProtocolMessageType,
    ) -> None:
        raise NotImplementedError()

    def eventHandle(
        self,
        _request: GeneratedProtocolMessageType
    ) -> str:
        raise NotImplementedError()

    def eventSink(
        self,
        _response: str
    ) -> GeneratedProtocolMessageType:
        raise NotImplementedError()


class RecognizerService(RecognizerServicer, SourceSinkService):
    async def Recognize(
        self,
        request: RecognizeRequest,
        _context: grpc.aio.ServicerContext,
    ) -> RecognizeResponse:
        """
        Send audio as bytes and receive the transcription of the audio.
        """
        logging.info(
            "Received request "
            f"[language={request.config.parameters.language}] "
            f"[sample_rate={request.config.parameters.sample_rate_hz}] "
            f"[topic={RecognitionResource.Model.Name(request.config.resource.topic)}]"
        )
        self.eventSource(request)
        response = self.eventHandle(request)
        return self.eventSink(response)

    def eventSource(
        self,
        request: RecognizeRequest,
    ) -> None:
        self._validateConfig(request.config)
        self._validateAudio(request.audio)

    def _validateConfig(
        self,
        config: RecognitionConfig,
    ) -> None:
        self._validateParameters(config.parameters)
        self._validateResource(config.resource)

    def _validateParameters(
        self,
        parameters: RecognitionParameters,
    ) -> None:
        if not Language.check(parameters.language):
            raise ValueError(f"Invalid value '{parameters.language}' for language parameter")
        if not SampleRate.check(parameters.sample_rate_hz):
            raise ValueError(f"Invalid value '{parameters.sample_rate_hz}' for sample_rate_hz parameter")

    def _validateResource(
        self,
        resource: RecognitionResource,
    ) -> None:
        try:
            RecognitionResource.Model.Name(resource.topic)
        except:
            raise ValueError(f"Invalid value '{resource.topic}' for topic resource")

    def _validateAudio(
        self,
        audio: bytes,
    ) -> None:
        if len(audio) == 0:
            raise ValueError(f"Empty value for audio")

    def eventHandle(
        self,
        request: RecognizeRequest
    ) -> str:
        language = Language.parse(request.config.parameters.language)
        if language == Language.EN_US:
            return "Hello, I am up and running. Received a message from you!"
        elif language == Language.ES_ES:
            return "Hola, estoy levantado y en marcha. ¡He recibido un mensaje tuyo!"
        elif language == Language.PT_BR:
            return "Olá, estou de pé, recebi uma mensagem sua!"

    def eventSink(
        self,
        response: str
    ) -> RecognizeResponse:
        result = {"text": response}
        return RecognizeResponse(**result)
