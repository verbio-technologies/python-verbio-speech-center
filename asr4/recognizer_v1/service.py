import abc
import grpc
import logging
import argparse
import math

from .runtime import OnnxRuntime, Session, OnnxSession, DecodingType

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



class RecognitionServiceConfiguration:
    def __init__(self, arguments: Optional[argparse.Namespace] = None):
        self.vocabulary = None
        self.formatterModelPath = None
        self.language = Language.EN_US
        self.model = None
        self.lmFile = None
        self.lexicon = None
        self.gpu = False
        self.numberOfWorkers = 1
        self.decodingType = DecodingType["GLOBAL"]
        self.lmAlgorithm = "viterbi"
        self.__setArguments(arguments)

    def __setArguments(self, arguments: argparse.Namespace):
        if arguments is not None:
            self.vocabulary = arguments.vocabulary
            self.formatterModelPath = arguments.formatter
            self.language = self._validateLanguage(arguments.language)
            self.model = arguments.model
            self.lexicon = arguments.lexicon
            self.lmFile = arguments.lm_model
            self.gpu = arguments.gpu
            self.numberOfWorkers = arguments.workers
            self.decodingType = DecodingType[
                getattr(arguments, "decoding_type", "GLOBAL")
            ]
            self.lmAlgorithm = arguments.lm_algorithm

    def createOnnxSession(self) -> OnnxSession:
        return OnnxSession(
            self.model,
            decoding_type=self.decodingType,
            providers=RecognitionServiceConfiguration._createProvidersList(self.gpu),
            number_of_workers=self.numberOfWorkers,
        )

    @staticmethod
    def _createProvidersList(gpu: bool) -> List[str]:
        providers = ["CPUExecutionProvider"]
        if gpu:
            providers = ["CUDAExecutionProvider"] + providers
        return providers

    @staticmethod
    def _validateLanguage(language: str) -> Language:
        if not Language.check(language):
            raise ValueError(f"Invalid language '{language}'")
        return Language.parse(language)


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
        configuration: RecognitionServiceConfiguration,
        formatter=None,
    ) -> None:
        self.logger = logging.getLogger("ASR4")
        self._language = configuration.language
        self._formatter = formatter
        self._runtime = self._createRuntime(
            configuration.createOnnxSession(),
            configuration.vocabulary,
            configuration.lmFile,
            configuration.lexicon,
            configuration.lmAlgorithm,
        )
        if formatter is None:
            self.logger.warning(
                "No formatter provided. Text will be generated without format"
            )

    @staticmethod
    def _createRuntime(
        session: Session,
        vocabularyPath: Optional[str],
        lmFile: Optional[str],
        lexicon: Optional[str],
        lmAlgorithm: Optional[str],
    ) -> OnnxRuntime:
        if vocabularyPath is not None:
            vocabulary = RecognizerService._readVocabulary(vocabularyPath)
            return OnnxRuntime(session, vocabulary, lmFile, lexicon, lmAlgorithm)
        else:
            return OnnxRuntime(session)

    @staticmethod
    def _readVocabulary(
        vocabularyPath: str,
    ) -> List[str]:
        with open(vocabularyPath) as f:
            vocabulary = f.read().splitlines()
        return vocabulary

    async def Recognize(
        self,
        request: RecognizeRequest,
        _context: grpc.aio.ServicerContext,
    ) -> RecognizeResponse:
        """
        Send audio as bytes and receive the transcription of the audio.
        """
        duration = self.calculateAudioDuration(request)
        self.logger.info(
            "Received request "
            f"[language={request.config.parameters.language}] "
            f"[sample_rate={request.config.parameters.sample_rate_hz}] "
            f"[length={duration.seconds}.%09d]"
            f"[topic={RecognitionResource.Model.Name(request.config.resource.topic)}]"
            % duration.nanos
        )
        self.eventSource(request)
        response = self.eventHandle(request)
        self.logger.info(f"Recognition result: '{response}'")
        return self.eventSink(response, duration)

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
        duration = self.calculateAudioDuration(innerRecognizeRequest)
        self.logger.debug(
            f" Processig audio with length [length={duration.seconds}.%09d]"
            % duration.nanos
        )
        response = self.eventHandle(innerRecognizeRequest)
        self.logger.info(f"Recognition result: '{response}'")
        innerRecognizeResponse = self.eventSink(response, duration)
        yield StreamingRecognizeResponse(
            results=StreamingRecognitionResult(
                alternatives=innerRecognizeResponse.alternatives,
                end_time=innerRecognizeResponse.end_time,
                duration=innerRecognizeResponse.duration,
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
        if self._formatter and words:
            self.logger.debug(f"Pre-formatter text: {words}")
            try:
                return " ".join(self._formatter.classify(words))
            except:
                self.logger.error(f"Error formatting sentence '{transcription}'")
                return " ".join(words)
        else:
            return " ".join(words)

    def eventSink(
        self, response: str, duration: Duration = Duration(seconds=0, nanos=0)
    ) -> RecognizeResponse:
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
            alternatives=[alternative],
            end_time=duration,
            duration=duration,
        )

    def calculateAudioDuration(self, request: RecognizeRequest) -> Duration:
        FRAME_SIZE = 2
        if request.config.parameters.audio_encoding == 0:
            FRAME_SIZE = 2
        NANOS_PER_SAMPLE = int(1_000_000_000 / request.config.parameters.sample_rate_hz)
        frames = len(request.audio) / FRAME_SIZE
        timeSec = int(math.floor(frames / request.config.parameters.sample_rate_hz))
        timeNanoSec = (int(frames) * NANOS_PER_SAMPLE) % 1_000_000_000
        return Duration(seconds=timeSec, nanos=timeNanoSec)
