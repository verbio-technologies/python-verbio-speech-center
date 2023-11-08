import toml
import grpc
import toml
import asyncio
from loguru import logger
from typing import Dict, AsyncIterator, Union

from .handler import EventHandler
from .types import RecognizerServicer
from .types import StreamingRecognizeRequest
from .types import StreamingRecognizeResponse

from asr4_engine import ASR4EngineFactory, ASR4Engine
from asr4_engine.data_classes import Language

_DEFAULT_ID = "unknown"


class RecognizerService(RecognizerServicer):
    def __init__(self, config: str) -> None:
        tomlConfiguration = toml.load(config)
        logger.debug(f"Toml configuration: {tomlConfiguration}")
        languageCode = tomlConfiguration.get("global", {}).get("language", "en-US")
        logger.info(f"Recognizer supported language is: {languageCode}")
        self._language = Language.parse(languageCode)
        self._engine = self._initializeEngine(tomlConfiguration, languageCode)

    def _initializeEngine(
        self,
        configuration: Dict[str, Dict[str, Union[str, float]]],
        languageCode: str,
    ) -> ASR4Engine:
        configuration["global"]["engine"] = configuration.get("global", {}).get(
            "engine", "w2v"
        )
        engine = ASR4EngineFactory().createEngine(
            config=toml.dumps(configuration), language=languageCode
        )
        return engine

    async def StreamingRecognize(
        self,
        request_iterator: AsyncIterator[StreamingRecognizeRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[StreamingRecognizeResponse]:
        """
        Send audio as a stream of bytes and receive the transcription of the audio through another stream.
        """
        metadata = self.__getContextMetadata(context)
        with logger.contextualize(
            user_id=metadata.get("user-id", _DEFAULT_ID),
            request_id=metadata.get("request-id", _DEFAULT_ID),
        ):
            handler = EventHandler(self._language, self._engine, context)
            listenerTask = asyncio.create_task(handler.listenForTranscription())
            try:
                async for request in request_iterator:
                    if listenerTask.done():
                        break
                    await handler.processStreamingRequest(request)
            except grpc.aio.AbortError as e:
                raise e
            except Exception as e:
                logger.error(e)
                await context.abort(grpc.StatusCode.INTERNAL, "Internal Server Error")
            finally:
                await self.__waitForListenerTask(handler, listenerTask)

        return

    def __getContextMetadata(self, context: grpc.aio.ServicerContext) -> Dict[str, str]:
        return dict(context.invocation_metadata())

    async def __waitForListenerTask(
        self, handler: EventHandler, listenerTask: asyncio.Task
    ):
        await handler.notifyEndOfAudio()
        await listenerTask
