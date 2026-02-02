import sys
sys.path.insert(1, '../proto/generated')
import pytest
from unittest.mock import Mock
from helpers.csr_client import CSRClient
from helpers.common import VerbioGrammar, RecognizerOptions
from concurrent.futures import ThreadPoolExecutor
import speechcenter.stt.recognition_streaming_response_pb2 as response


def test_recognition_full_flow():
    mock_stub = Mock()
    options = RecognizerOptions()
    options.inactivity_timeout = 0.1
    options.asr_version = "V2"
    options.topic = "GENERIC"
    options.language = "en-US"
    options.label = "label"
    options.formatting = False
    options.diarization = False

    audio_resource = Mock()
    audio_resource.sample_rate = 16000
    audio_resource.audio = b'0000000000000000'

    executor = ThreadPoolExecutor()
    recognition_result = response.RecognitionResult(is_final=True)
    mock_response = response.RecognitionStreamingResponse(result=recognition_result)
    mock_stub.StreamingRecognize.return_value = [mock_response, mock_response]
    client = CSRClient(executor, mock_stub, options, audio_resource, "token")
    client.send_audio()
    client.wait_for_response()


def test_recognition_full_flow_exception():
    mock_stub = Mock()
    options = RecognizerOptions()
    options.inactivity_timeout = 0.1
    options.asr_version = "V1"
    executor = ThreadPoolExecutor()
    audio_resource = Mock()
    audio_resource.sample_rate = 8000
    mock_stub.StreamingRecognize.side_effect = Exception("Exception while sending audio")
    client = CSRClient(executor, mock_stub, options, audio_resource, "token")
    with pytest.raises(Exception):
        client.send_audio()
        client.wait_for_response()


def test_recognition_full_flow_grammar():
    mock_stub = Mock()
    options = RecognizerOptions()
    options.inactivity_timeout = 0.1
    options.asr_version = "V2"
    options.grammar = VerbioGrammar(VerbioGrammar.URI, "test/grammar")
    options.language = "en-US"
    options.label = "label"
    options.formatting = False
    options.diarization = False

    audio_resource = Mock()
    audio_resource.sample_rate = 16000
    audio_resource.audio = b'0000000000000000'

    executor = ThreadPoolExecutor()
    recognition_result = response.RecognitionResult(is_final=True)
    mock_response = response.RecognitionStreamingResponse(result=recognition_result)
    mock_stub.StreamingRecognize.return_value = [mock_response, mock_response]
    client = CSRClient(executor, mock_stub, options, audio_resource, "token")
    client.send_audio()
    client.wait_for_response()
