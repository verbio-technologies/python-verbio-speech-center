import abc
import grpc
import logging
import pyformatter

from .runtime import OnnxRuntime, Session

from asr4.types.language import Language
from .types import RecognizerServicer
from .types import RecognizeRequest
from .types import StreamingRecognizeRequest
from .types import RecognitionConfig
from .types import RecognitionParameters
from .types import RecognitionResource
from .types import RecognizeResponse
from .types import StreamingRecognizeResponse
from .types import StreamingRecognitionResult
from .types import RecognitionAlternative
from .types import Duration
from .types import WordInfo
from .types import SampleRate
from .types import AudioEncoding

from typing import Optional, List
from google.protobuf.reflection import GeneratedProtocolMessageType


class SourceSinkService(abc.ABC):
    def eventSource(
        self,
        _request: GeneratedProtocolMessageType,
    ) -> None:
        raise NotImplementedError()

    def eventHandle(self, _request: GeneratedProtocolMessageType) -> str:
        raise NotImplementedError()

    def eventSink(self, _response: str) -> GeneratedProtocolMessageType:
        raise NotImplementedError()


class RecognizerService(RecognizerServicer, SourceSinkService):
    def __init__(
        self,
        session: Session,
        language: Language = Language.EN_US,
        vocabularyPath: Optional[str] = None,
        formatterPath: Optional[str] = None,
    ) -> None:
        self._language = language
        self._createRuntime(session, vocabularyPath)
        self._createFormatter(formatterPath)
        self.logger = logging.getLogger("ASR4")

    def _createRuntime(
        self,
        session: Session,
        vocabularyPath: Optional[str],
    ):
        if vocabularyPath != None:
            vocabulary = self._readVocabulary(vocabularyPath)
            self._runtime = OnnxRuntime(session, vocabulary)
        else:
            self._runtime = OnnxRuntime(session)

    def _readVocabulary(
        self,
        vocabularyPath: str,
    ) -> List[str]:
        with open(vocabularyPath) as f:
            vocabulary = f.read().splitlines()
        return vocabulary

    def _createFormatter(
        self,
        path: Optional[str],
    ):
        self._formatter = None
        if path != None:
            self._formatter = pyformatter.PyFormatter(
                self._language.asFormatter(), path, b"", b"", dict()
            )

    async def Recognize(
        self,
        request: RecognizeRequest,
        _context: grpc.aio.ServicerContext,
    ) -> RecognizeResponse:
        """
        Send audio as bytes and receive the transcription of the audio.
        """
        self.logger.info(
            "Received request "
            f"[language={request.config.parameters.language}] "
            f"[sample_rate={request.config.parameters.sample_rate_hz}] "
            f"[topic={RecognitionResource.Model.Name(request.config.resource.topic)}]"
        )
        self.eventSource(request)
        response = self.eventHandle(request)
        self.logger.info(f"Recognition result: '{response}'")
        return self.eventSink(response)

    async def StreamingRecognize(
        self,
        request_iterator: StreamingRecognizeRequest,
        _context: grpc.aio.ServicerContext,
    ) -> StreamingRecognizeResponse:
        """
        Send audio as a stream of bytes and receive the transcription of the audio through another stream.
        """
        innerRecognizeRequest = RecognizeRequest()
        async for request in request_iterator:
            if request.HasField("config"):
                self.logger.info(
                    "Received streaming request "
                    f"[language={request.config.parameters.language}] "
                    f"[sample_rate={request.config.parameters.sample_rate_hz}] "
                    f"[topic={RecognitionResource.Model.Name(request.config.resource.topic)}]"
                )
                innerRecognizeRequest.config.CopyFrom(request.config)
            if request.HasField("audio"):
                innerRecognizeRequest.audio = (
                    innerRecognizeRequest.audio + request.audio
                )
        self.eventSource(innerRecognizeRequest)
        self.logger.debug(
            f" Processig audio with length %d" % len(innerRecognizeRequest.audio)
        )
        response = self.eventHandle(innerRecognizeRequest)
        self.logger.info(f"Recognition result: '{response}'")
        innerRecognizeResponse = self.eventSink(response)
        yield StreamingRecognizeResponse(
            results=StreamingRecognitionResult(
                alternatives=innerRecognizeResponse.alternatives,
                end_time=innerRecognizeResponse.end_time,
                is_final=True,
            )
        )

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
            raise ValueError(
                f"Invalid value '{parameters.language}' for language parameter"
            )
        if not SampleRate.check(parameters.sample_rate_hz):
            raise ValueError(
                f"Invalid value '{parameters.sample_rate_hz}' for sample_rate_hz parameter"
            )
        if not AudioEncoding.check(parameters.audio_encoding):
            raise ValueError(
                f"Invalid value '{parameters.audio_encoding}' for audio_encoding parameter"
            )

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

    def eventHandle(self, request: RecognizeRequest) -> str:
        transcription = self._runRecognition(request)
        return self._formatWords(transcription)

    def _runRecognition(self, request: RecognizeRequest) -> str:
        language = Language.parse(request.config.parameters.language)
        sample_rate_hz = request.config.parameters.sample_rate_hz
        if language == self._language:
            return self._runtime.run(request.audio, sample_rate_hz).sequence
        else:
            raise ValueError(
                f"Invalid language '{language}'. Only '{self._language}' is supported."
            )

    def _formatWords(self, transcription: str) -> str:
        words = list(filter(lambda x: len(x) > 0, transcription.split(" ")))
        if self._formatter:
            self.logger.debug(f"Pre-formatter text: {words}")
            return " ".join(self._formatter.classify(words))
        else:
            return " ".join(words)

    def eventSink(self, response: str) -> RecognizeResponse:
        words = map(
            lambda token: WordInfo(
                start_time=Duration(seconds=0, nanos=0),
                end_time=Duration(seconds=0, nanos=0),
                word=token,
                confidence=1.0,
            ),
            response.split(" "),
        )
        alternative = RecognitionAlternative(
            transcript=response, confidence=1.0, words=words
        )
        return RecognizeResponse(
            alternatives=[alternative], end_time=Duration(seconds=0, nanos=0)
        )
