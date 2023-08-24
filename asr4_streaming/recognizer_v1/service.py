import abc
import grpc
import logging
import argparse
from datetime import timedelta
from dataclasses import dataclass, field

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

import toml
import numpy as np

from asr4_engine.data_classes import Signal, Segment
from asr4_engine.data_classes.transcription import WordTiming
from asr4.engines.wav2vec import Wav2VecEngineFactory
from asr4.engines.wav2vec.v1.engine_types import Language


@dataclass
class TranscriptionResult:
    transcription: str
    score: float
    words: List[WordTiming]


class RecognitionServiceConfiguration:
    def __init__(self, arguments: Optional[argparse.Namespace] = None):
        self.config = None
        self.__setArguments(arguments)

    def __setArguments(self, arguments: argparse.Namespace):
        if arguments is not None:
            self.config = arguments.config

    def initializeEngine(self, tomlConfiguration: dict, languageCode: str) -> Wav2VecEngineFactory:
        factory = Wav2VecEngineFactory()
        engine = factory.create_engine()
        engine.initialize(
            config=toml.dumps(tomlConfiguration), language=languageCode
        )
        return engine

    @staticmethod
    def _initializeEnginersList(gpu: bool) -> List[str]:
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
    def __init__(self, configuration: RecognitionServiceConfiguration) -> None:
        self.logger = logging.getLogger("ASR4")
        tomlConfiguration = toml.load(configuration.config)
        logging.debug(f"Toml configuration file: {configuration.config}")
        logging.debug(f"Toml configuration: {tomlConfiguration}")
        self._languageCode = tomlConfiguration.get("global", {}).get(
            "language", "en-US"
        )
        self._language = Language.parse(self._languageCode)
        self._engine = configuration.initializeEngine(tomlConfiguration, self._languageCode)
        logging.info(f"Recognizer supported language is: {self._languageCode}")


    async def Recognize(
        self,
        request: RecognizeRequest,
        _context: grpc.aio.ServicerContext,
    ) -> RecognizeResponse:
        """
        Send audio as bytes and receive the transcription of the audio.
        """
        self.eventSource(request)
        duration = self.calculateAudioDuration(request)
        self.logger.info(
            "Received request "
            f"[language={request.config.parameters.language}] "
            f"[sample_rate={request.config.parameters.sample_rate_hz}] "
            f"[formatting={request.config.parameters.enable_formatting}] "
            f"[length={len(request.audio)}] "
            f"[duration={duration.ToTimedelta().total_seconds()}] "
            f"[topic={RecognitionResource.Model.Name(request.config.resource.topic)}]"
        )
        response = self.eventHandle(request)
        response = self.eventSink(response, duration, duration)
        self.logger.info(f"Recognition result: '{response.alternatives[0].transcript}'")
        return response

    async def StreamingRecognize(
        self,
        request_iterator: StreamingRecognizeRequest,
        _context: grpc.aio.ServicerContext,
    ) -> StreamingRecognizeResponse:
        """
        Send audio as a stream of bytes and receive the transcription of the audio through another stream.
        """
        innerRecognizeRequest, totalDuration = RecognizeRequest(), Duration()
        audio = bytes(0)

        async for request in request_iterator:
            if request.HasField("config"):
                self.logger.info(
                    "Received streaming request "
                    f"[language={request.config.parameters.language}] "
                    f"[sample_rate={request.config.parameters.sample_rate_hz}] "
                    f"[formatting={request.config.parameters.enable_formatting}] "
                    f"[topic={RecognitionResource.Model.Name(request.config.resource.topic)}]"
                )
                innerRecognizeRequest.config.CopyFrom(request.config)
            if request.HasField("audio"):
                audio += request.audio
                self.logger.info(
                    f"Received partial audio " f"[length={len(request.audio)}] "
                )

        innerRecognizeRequest.audio = audio
        self.eventSource(innerRecognizeRequest)
        duration = self.calculateAudioDuration(innerRecognizeRequest)
        self.logger.info(
            f"Received total audio "
            f"[length={len(request.audio)}] "
            f"[duration={duration.ToTimedelta().total_seconds()}] "
        )
        totalDuration = RecognizerService.addAudioDuration(totalDuration, duration)
        response = self.eventHandle(innerRecognizeRequest)
        innerRecognizeResponse = self.eventSink(response, duration, totalDuration)
        self.logger.info(
            f"Recognition result: '{innerRecognizeResponse.alternatives[0].transcript}'"
        )
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

    def eventHandle(self, request: RecognizeRequest) -> TranscriptionResult:
        language = Language.parse(request.config.parameters.language)
        sample_rate_hz = request.config.parameters.sample_rate_hz
        if language == self._language:
            result = self._engine.recognize(
                Signal(np.frombuffer(request.audio, dtype=np.int16), sample_rate_hz),
                language=self._languageCode,
            )

            return TranscriptionResult(
                transcription=result.text,
                score=self.calculateAverageScore(result.segments),
                words=self.extractWords(result.segments),
            )

        else:
            raise ValueError(
                f"Invalid language '{language}'. Only '{self._language}' is supported."
            )

    def calculateAverageScore(self, segments: List[Segment]) -> float:
        acummScore = 0.0
        for segment in segments:
            acummScore += segment.avg_logprob
        return acummScore / len(segments)

    def extractWords(self, segments: List[Segment]) -> List[WordTiming]:
        words = []
        for segment in segments:
            words.extend(segment.words)
        return words

    def eventSink(
        self,
        response: TranscriptionResult,
        duration: Duration = Duration(seconds=0, nanos=0),
        endTime: Duration = Duration(seconds=0, nanos=0),
    ) -> RecognizeResponse:
        def getWord(word: WordTiming) -> WordInfo:
            wordInfo = WordInfo(
                start_time=Duration(),
                end_time=Duration(),
                word=word.word,
                confidence=word.probability,
            )
            wordInfo.start_time.FromTimedelta(td=timedelta(seconds=word.start))
            wordInfo.end_time.FromTimedelta(td=timedelta(seconds=word.end))
            return wordInfo

        if len(response.words) > 0:
            words = [getWord(word) for word in response.words]
        else:
            words = []

        alternative = RecognitionAlternative(
            transcript=response.transcription, confidence=response.score, words=words
        )
        return RecognizeResponse(
            alternatives=[alternative],
            end_time=endTime,
            duration=duration,
        )

    def calculateAudioDuration(self, request: RecognizeRequest) -> Duration:
        duration = Duration()
        audioEncoding = AudioEncoding.parse(request.config.parameters.audio_encoding)
        # We only support 1 channel
        bytesPerFrame = audioEncoding.getSampleSizeInBytes() * 1
        framesNumber = len(request.audio) / bytesPerFrame
        td = timedelta(
            seconds=(framesNumber / request.config.parameters.sample_rate_hz)
        )
        duration.FromTimedelta(td=td)
        return duration

    @staticmethod
    def addAudioDuration(a: Duration, b: Duration) -> Duration:
        duration = Duration()
        total = a.ToTimedelta().total_seconds() + b.ToTimedelta().total_seconds()
        duration.FromTimedelta(td=timedelta(seconds=total))
        return duration
