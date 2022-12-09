import os
import re
import sys
import jiwer
import pytest
import unittest
from shutil import rmtree
from subprocess import Popen, PIPE
from typing import Optional
from asr4.recognizer import Language


class TestRecognizerUtils(object):
    def readReference(self, referencePath: str) -> str:
        with open(referencePath) as f:
            return " ".join(f.read().splitlines())

    def _runRecognition(
        self,
        language: str,
        *,
        audioPath: Optional[str] = None,
        guiPath: Optional[str] = None,
        output: Optional[str] = None,
    ) -> Popen:
        cmd = [
            "python3",
            f"{self.rootdir}/bin/client.py",
            "-v",
            "INFO",
            "--language",
            language,
            "--host",
            self._host,
        ]

        if audioPath:
            cmd.extend(["--audio-path", audioPath])
        elif guiPath:
            cmd.extend(["--gui-path", guiPath])
        if output:
            cmd.extend(["--metrics", "--output", output])

        return Popen(
            cmd,
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
        )

    def launchRecognitionProcess(self, audioPath: str, language: str) -> Popen:
        return self._runRecognition(language, audioPath=audioPath)

    def runGuiRecognition(self, guiPath: str, language: str) -> Popen:
        return self._runRecognition(language, guiPath=guiPath)

    def runRecognitionWithMetrics(
        self, audioPath: str, language: str, output: str
    ) -> Popen:
        return self._runRecognition(language, audioPath=audioPath, output=output)

    def runGuiRecognitionWithMetrics(
        self, guiPath: str, language: str, output: str
    ) -> Popen:
        return self._runRecognition(language, guiPath=guiPath, output=output)

    @staticmethod
    def checkStatus(status: int, stderr: str) -> None:
        try:
            assert status == 0
        except:
            print(stderr, file=sys.stderr)
            exit(-1)

    @staticmethod
    def evaluateHypothesis(reference: str, hypothesis: str) -> None:
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

    def removeOutputContents(self) -> None:
        for _, dirs, files in os.walk(self._output):
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
        self._hostName = os.getenv("ASR4_HOSTNAME", "0.0.0.0")
        self._hostPort = os.getenv("ASR4_PORT", 50051)
        self._host = f"{self._hostName}:{self._hostPort}"
        self._audio = f"{os.path.join(self.datadir, self._language)}-1.wav"
        self._audio8k = f"{os.path.join(self.datadir, self._language)}.8k.wav"
        self._audio16k = f"{os.path.join(self.datadir, self._language)}.16k.wav"
        self._gui = f"{os.path.join(self.datadir, self._language)}.gui"
        referencePath = f"{os.path.join(self.datadir, self._language)}-1.txt"
        self._reference = self.readReference(referencePath)
        self._output = self.datadir + "/output"

    def testRecognizeRequest(self):
        hypothesis = self._recognizeAudio(self._audio, self._language)
        self.assertGreater(len(hypothesis), 1)
        self.evaluateHypothesis(self._reference, hypothesis)

    def testRecognitionAudioDifferentSampleRate(self):
        self.assertEqual(
            self._recognizeAudio(self._audio8k, self._language),
            self._recognizeAudio(self._audio16k, self._language),
        )

    def testRecognizeGuiRequest(self):
        process = self.runGuiRecognition(self._gui, self._language)
        status = process.wait(timeout=900)
        self.checkStatus(status, process.stderr.read())

        output = process.stdout.read()
        hypothesis = re.findall('RecognizeRequest first alternative: "(.+?)"', output)

        self.assertEqual(len(hypothesis), 3)
        self.assertGreater(len(hypothesis[0]), 0)
        self.assertGreater(len(hypothesis[1]), 0)
        self.assertGreater(len(hypothesis[2]), 0)

    def testRecognizeRequestWithOtherLanguages(self):
        currentLanguage = Language.parse(self._language)
        for otherLanguage in Language:
            if otherLanguage != currentLanguage:
                process = self.launchRecognitionProcess(
                    self._audio, otherLanguage.value
                )
                _status = process.wait(timeout=900)
                output = process.stdout.read()
                match = re.search(
                    f"Invalid language '{otherLanguage}'. Only '{currentLanguage}' is supported.",
                    output,
                )
                self.assertIsNotNone(match)

    def testEmptyRecognizeRequest(self):
        process = self.launchRecognitionProcess(
            f"{self.datadir}/empty.wav", self._language
        )
        status = process.wait(timeout=900)
        self.assertEqual(status, 1)

    def testEmptyGuiRecognizeRequest(self):
        process = self.runGuiRecognition(f"{self.datadir}/empty.gui", self._language)
        status = process.wait(timeout=60)
        self.assertEqual(status, 0)

    def testGuiEvaluationResultsExistInPath(self):
        process = self.runGuiRecognitionWithMetrics(
            self._gui, self._language, self._output
        )
        status = process.wait(timeout=900)
        self.checkStatus(status, process.stderr.read())

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
        process = self.runRecognitionWithMetrics(
            self._audio, self._language, self._output
        )
        status = process.wait(timeout=900)
        self.checkStatus(status, process.stderr.read())
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

    def _recognizeAudio(self, audio, language):
        process = self.launchRecognitionProcess(audio, language)
        status = process.wait(timeout=900)
        self.checkStatus(status, process.stderr.read())
        output = process.stdout.read()
        match = re.search('RecognizeRequest first alternative: "(.+?)"', output)
        return (
            match.group(match.lastindex)
            if match is not None and match.lastindex is not None
            else ""
        )
