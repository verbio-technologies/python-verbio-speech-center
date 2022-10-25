import os
import re
import sys
import jiwer
import pytest
import unittest
from subprocess import Popen, PIPE
import time

from asr4.recognizer import Language


class TestRecognizerUtils(object):
    def readReference(self, referencePath: str) -> str:
        with open(referencePath) as f:
            return " ".join(f.read().splitlines())

    def runRecognition(self, audioPath: str, language: str) -> Popen:
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
            ],
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
        )

    def runGuiRecognition(self, guiPath: str, language: str) -> Popen:
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
            ],
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
        )

    def runGuiRecognitionWithTestPath(
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
                "--output-dir",
                output,
            ],
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
        )

    def runGuiRecognitionWithTestPathDefault(
        self, guiPath: str, language: str
    ) -> Popen:
        return Popen(
            [
                "python3",
                f"{self.rootdir}/bin/client.py",
                "--gui-path",
                guiPath,
                "--language",
                language,
                "--host",
                self._host,
                "--metrics",
            ],
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
        )

    def checkStatus(self, status: int, stderr: str) -> None:
        try:
            assert status == 0
        except:
            print(stderr, file=sys.stderr)
            exit(-1)

    def evaluateHypothesis(self, reference: str, hypothesis: str) -> None:
        measures = jiwer.compute_measures(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        print("\nHypothesis =", hypothesis)
        print("\nReference =", reference)
        print("\nCharacter Error Rate (CER) =", round(cer, 3))
        print("Word Error Rate (WER) =", round(measures["wer"], 3))
        print("Match Error Rate (MER) =", round(measures["mer"], 3))
        print("Word Information Preserved (WIP) =", round(measures["wip"], 3))
        print("Word Information Lost (WIL) =", round(measures["wil"], 3))
        print("Correct =", measures["hits"])
        print("Substitutions =", measures["substitutions"])
        print("Deletions =", measures["deletions"])
        print("Insertions =", measures["insertions"])


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

    def testRecognizeRequest(self):
        process = self.runRecognition(self._audio, self._language)
        status = process.wait(timeout=100)
        self.checkStatus(status, process.stderr.read())
        output = process.stdout.read()
        match = re.search('RecognizeRequest text: "(.+?)"', output)
        hypothesis = match.group(match.lastindex)

        assert len(hypothesis) > 1

        if match != None and match.lastindex != None:
            self.evaluateHypothesis(self._reference, hypothesis)

    def testRecognizeGuiRequest(self):
        process = self.runGuiRecognition(self._gui, self._language)
        status = process.wait(timeout=600)
        self.checkStatus(status, process.stderr.read())

        output = process.stdout.read()
        hypothesis = re.findall('RecognizeRequest text: "(.+?)"', output)

        assert len(hypothesis) == 3
        assert len(hypothesis[0]) > 0
        assert len(hypothesis[1]) > 0
        assert len(hypothesis[2]) > 0

    def testRecognizeRequestWithOtherLanguages(self):
        currentLanguage = Language.parse(self._language)
        for otherLanguage in Language:
            if otherLanguage != currentLanguage:
                process = self.runRecognition(self._audio, otherLanguage.value)
                _status = process.wait(timeout=100)
                output = process.stdout.read()
                match = re.search(
                    f"Invalid language '{otherLanguage}'. Only '{currentLanguage}' is supported.",
                    output,
                )
                self.assertIsNotNone(match)

    def testEmptyRecognizeRequest(self):
        process = self.runRecognition(f"{self.datadir}/empty.wav", self._language)
        status = process.wait(timeout=100)
        self.assertEqual(status, 1)

    def testEmptyGuiRecognizeRequest(self):
        process = self.runGuiRecognition(f"{self.datadir}/empty.gui", self._language)
        status = process.wait(timeout=60)
        self.assertEqual(status, 0)

    def testGuiEvaluationResultsExistInPath(self):
        self.runGuiRecognitionWithTestPath(self._gui, self._language, self._output)
        time.sleep(40)
        assert os.path.exists(
            f"{self._output }/trnHypothesis.trn"
        ), "trnHypothesis does not exists"
        assert os.path.exists(
            f"{self._output }/trnReferences.trn"
        ), "trnReferences does not exists"
        assert os.path.exists(
            f"{self._output }/wer/id.pra.analysis"
        ), "analysis file does not exists"
        assert os.path.exists(
            f"{self._output }/wer/id.dtl"
        ), "analysis file does not exists"
        assert os.path.exists(
            f"{self._output }/id_results.json"
        ), "analysis file does not exists"
