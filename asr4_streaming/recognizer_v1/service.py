import toml
import grpc
import toml
import logging
import asyncio
from asyncio import Task
from typing import Dict, AsyncIterator, Union, Optional

from .handler import EventHandler
from .types import RecognizerServicer
from .types import StreamingRecognizeRequest
from .types import StreamingRecognizeResponse

from asr4.engines.wav2vec import Wav2VecEngineFactory
from asr4.engines.wav2vec.wav2vec_engine import Wav2VecEngine
from asr4.engines.wav2vec.v1.engine_types import Language


class RecognizerService(RecognizerServicer):
    def __init__(self, config: str) -> None:
        self._logger = logging.getLogger("ASR4")
        tomlConfiguration = toml.load(config)
        self._logger.debug(f"Toml configuration: {tomlConfiguration}")
        languageCode = tomlConfiguration.get("global", {}).get("language", "en-US")
        self._logger.info(f"Recognizer supported language is: {languageCode}")
        self._language = Language.parse(languageCode)
        self._engine = self._initializeEngine(tomlConfiguration, languageCode)

    def _initializeEngine(
        self,
        tomlConfiguration: Dict[str, Dict[str, Union[str, float]]],
        languageCode: str,
    ) -> Wav2VecEngine:
        engine = Wav2VecEngineFactory().create_engine()
        engine.initialize(config=toml.dumps(tomlConfiguration), language=languageCode)
        return engine

    async def StreamingRecognize(
        self,
        request_iterator: AsyncIterator[StreamingRecognizeRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[StreamingRecognizeResponse]:
        """
        Send audio as a stream of bytes and receive the transcription of the audio through another stream.
        """
        handler = EventHandler(self._language, self._engine, context)
        listenerTask = asyncio.create_task(handler.listenForTranscription())
        async for request in request_iterator:
            await handler.source(request)
        await handler.notifyEndOfAudio()
        await listenerTask
        return
