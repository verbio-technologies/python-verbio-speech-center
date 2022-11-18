import os
import pytest
import unittest
from subprocess import Popen, PIPE
import time
from shutil import rmtree

from asr4.recognizer import Language


class TestRecognizerUtils(object):
    def readReference(self, referencePath: str) -> str:
        with open(referencePath) as f:
            return " ".join(f.read().splitlines())

    def runRecognitionWithMetrics(
        self, audioPath: str, language: str, output: str
    ) -> Popen:
        return Popen(
            [
                "python",
                f"{self.rootdir}/bin/client.py",
                "--audio-path",
                audioPath,
                "--language",
                language,
                "--host",
                self._host,
                "--metrics",
                "--output",
                output,
            ],
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
        )

    def runGuiRecognitionWithMetrics(
        self, guiPath: str, language: str, output: str
    ) -> Popen:
        return Popen(
            [
                "python",
                f"{self.rootdir}/bin/client.py",
                "--gui-path",
                guiPath,
                "--language",
                language,
                "--host",
                self._host,
                "--metrics",
                "--output",
                output,
            ],
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
        )

    def removeOutputContents(self) -> None:
        for root, dirs, files in os.walk(self._output):
            for f in files:
                os.remove(os.path.join(self._output, f))
            for d in dirs:
                rmtree(os.path.join(self._output, d))


@pytest.mark.usefixtures("datadir")
class TestRecognizerService(unittest.TestCase, TestRecognizerUtils):
    @pytest.fixture(autouse=True)
    def rootdir(self, pytestconfig):
        self.rootdir = str(pytestconfig.rootdir)

    @pytest.fixture(autouse=True)
    def datadir(self, pytestconfig):
        self.datadir = f"{pytestconfig.rootdir}/tests/integration/data"

    def setUp(self) -> None:
        self._language = os.getenv("LANGUAGE", "en-us")
        self._hostName = os.getenv("ASR4_HOSTNAME", "asr4-server")
        self._hostPort = os.getenv("ASR4_PORT", 50051)
        self._host = f"{self._hostName}:{self._hostPort}"
        self._audio = f"{os.path.join(self.datadir, self._language)}-1.wav"
        self._gui = f"{os.path.join(self.datadir, self._language)}.gui"
        referencePath = f"{os.path.join(self.datadir, self._language)}-1.txt"
        self._reference = self.readReference(referencePath)
        self._output = self.datadir + "/output"

    def testGuiEvaluationResultsExistInPath(self):
        process = self.runGuiRecognitionWithMetrics(
            self._gui, self._language, self._output
        )
        status = process.wait(timeout=4000)
        assert os.path.exists(
            f"{self._output }/trnHypothesis.trn"
        ), "trnHypothesis does not exist"
        assert os.path.exists(
            f"{self._output }/trnReferences.trn"
        ), "trnReferences does not exist"
        assert os.path.exists(
            f"{self._output }/wer/test_{self._language}.pra.analysis"
        ), "analysis file does not exist"
        assert os.path.exists(
            f"{self._output }/wer/test_{self._language}.dtl"
        ), "analysis file does not exist"
        assert os.path.exists(
            f"{self._output }/test_{self._language}_results.tsv"
        ), "analysis file does not exist"
        self.removeOutputContents()

    def testAudioEvaluationResultsExistInPath(self):
        self.runRecognitionWithMetrics(self._audio, self._language, self._output)
        time.sleep(40)
        assert os.path.exists(
            f"{self._output }/trnHypothesis.trn"
        ), "trnHypothesis does not exist"
        assert os.path.exists(
            f"{self._output }/trnReferences.trn"
        ), "trnReferences does not exist"
        assert os.path.exists(
            f"{self._output }/wer/test_{self._language}.pra.analysis"
        ), "analysis file does not exist"
        assert os.path.exists(
            f"{self._output }/wer/test_{self._language}.dtl"
        ), "analysis file does not exist"
        assert os.path.exists(
            f"{self._output }/test_{self._language}_results.tsv"
        ), "analysis file does not exist"
        self.removeOutputContents()
