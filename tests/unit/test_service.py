import unittest
import pytest
from unittest.mock import patch, Mock
from grpc.aio import ServicerContext
import os

from tests.unit.test_event_handler import initializeMockEngine, initializeMockContext
from asr4.engines.wav2vec.wav2vec_engine import Wav2VecEngine
from asr4_streaming.recognizer import RecognizerService


@pytest.mark.usefixtures("datadir")
class TestService(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def datadir(self, pytestconfig):
        self.datadir = f"{pytestconfig.rootdir}/tests/functional/data"

    @patch("asr4_streaming.recognizer.RecognizerService._initializeEngine")
    def testGetContextMetadata(self, mock):
        mock.return_value = initializeMockEngine(Mock(Wav2VecEngine), language="en-US")
        mockContext = initializeMockContext(Mock(ServicerContext))
        service = RecognizerService(
            os.path.join(self.datadir, "asr4_streaming_config_en-us.toml")
        )
        self.assertEqual(
            service.getContextMetadata(mockContext),
            {"user-id": "testUser", "request-id": "testRequest"},
        )
