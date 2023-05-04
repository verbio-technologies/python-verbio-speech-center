import os
import re
import sys
import pytest
import unittest
from subprocess import Popen, PIPE
from typing import Optional


class TestFormatterUtils(object):
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
            "TRACE",
            "--format",
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

    def _runNoFormatRecognition(
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
            "TRACE",
            "--no-format",
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

    def launchRecognitionWithNoFormatting(self, audioPath: str, language: str) -> Popen:
        return self._runNoFormatRecognition(language, audioPath=audioPath)

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


@pytest.mark.usefixtures("datadir")
class TestFormatter(unittest.TestCase, TestFormatterUtils):
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
        self._audio = f"{os.path.join(self.datadir, self._language)}-fmt.wav"
        self._output = self.datadir + "/output"

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

    def testRecognizeRequestNoFormattedEN_US(self):
        process = self.launchRecognitionWithNoFormatting(self._audio, self._language)
        status = process.wait(timeout=900)
        self.checkStatus(status, process.stderr.read())
        output = process.stdout.read()
        match = re.search('RecognizeRequest first alternative: "(.+?)"', output)
        hypothesis = (
            match.group(match.lastindex)
            if match is not None and match.lastindex is not None
            else ""
        )
        self.assertEqual(
            hypothesis,
            "what is the cost of flight cee o one six three one from nashville to chicago",
        )

    def testRecognizeRequestFormattedEN_US(self):
        process = self.launchRecognitionProcess(self._audio, self._language)
        status = process.wait(timeout=900)
        self.checkStatus(status, process.stderr.read())
        output = process.stdout.read()
        match = re.search('RecognizeRequest first alternative: "(.+?)"', output)
        hypothesis = (
            match.group(match.lastindex)
            if match is not None and match.lastindex is not None
            else ""
        )
        self.assertEqual(
            hypothesis,
            "What is the cost of flight? Cee O 1 6 3 1 from Nashville to Chicago,",
        )

    def testRecognizeRequestNoFormattedES(self):
        process = self.launchRecognitionWithNoFormatting(self._audio, self._language)
        status = process.wait(timeout=900)
        self.checkStatus(status, process.stderr.read())
        output = process.stdout.read()
        match = re.search('RecognizeRequest first alternative: "(.+?)"', output)
        hypothesis = (
            match.group(match.lastindex)
            if match is not None and match.lastindex is not None
            else ""
        )
        self.assertEqual(
            hypothesis,
            "erre cuatrocientos setenta y cinco doscientos senchenta y cuatro tres e",
        )

    def testRecognizeRequestFormattedES(self):
        process = self.launchRecognitionProcess(self._audio, self._language)
        status = process.wait(timeout=900)
        self.checkStatus(status, process.stderr.read())
        output = process.stdout.read()
        match = re.search('RecognizeRequest first alternative: "(.+?)"', output)
        hypothesis = (
            match.group(match.lastindex)
            if match is not None and match.lastindex is not None
            else ""
        )
        self.assertEqual(hypothesis, "R. 2730762-H")
