import sys
sys.path.insert(1, '../proto/generated')

from unittest.mock import Mock
from helpers.csr_client import CSRClient
from helpers.common import RecognizerOptions
from concurrent.futures import ThreadPoolExecutor
import recognition_streaming_response_pb2 as response


def test_recognition_full_flow():
    mock_stub = Mock()
    options = RecognizerOptions()
    options.inactivity_timeout = 0.1
    executor = ThreadPoolExecutor()
    audio_resource = Mock()
    recognition_result = response.RecognitionResult(is_final=True)
    mock_response = response.RecognitionStreamingResponse(result=recognition_result)
    mock_stub.StreamingRecognize.return_value = [mock_response, mock_response]
    client = CSRClient(executor, mock_stub, options, audio_resource, "token")
    client.send_audio()
    client.wait_for_response()
