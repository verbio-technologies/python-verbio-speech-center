import json, os, re, sys
import jiwer
import pytest
import unittest
from shutil import rmtree
from subprocess import Popen, PIPE
from typing import Optional

from asr4.engines.wav2vec.v1.engine_types import Language


class TimeStampsStatistics:
    numberOfWords: [int] = 0.0
    numberOfSilences: [int] = 0.0
    speechTime: [float] = 0.0
    meanWordDuration: [float] = 0.0
    maxWordDuration: [float] = sys.float_info.min
    minWordDuration: [float] = sys.float_info.max
    silenceTime: [float] = 0.0
    minSilenceDuration: [float] = sys.float_info.max
    maxSilenceDuration: [float] = sys.float_info.min
    meanSilenceDuration: [float] = 0.0

    def updateSpeechStats(self, wordDuration):
        self.speechTime += wordDuration
        self.numberOfWords += 1
        self.meanWordDuration = self.speechTime / self.numberOfWords
        if wordDuration > self.maxWordDuration:
            self.maxWordDuration = wordDuration
        if wordDuration < self.minWordDuration:
            self.minWordDuration = wordDuration

    def updateSilenceStats(self, silenceDuration):
        self.silenceTime += silenceDuration
        self.numberOfSilences += 1
        self.meanSilenceDuration = self.silenceTime / self.numberOfSilences
        if silenceDuration < self.minSilenceDuration:
            self.minSilenceDuration = silenceDuration
        if silenceDuration > self.maxSilenceDuration:
            self.maxSilenceDuration = silenceDuration


def parseSeconds(text: str) -> float:
    return float(text[:-1])


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
            "TRACE",
            "--json",
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
        self._gui = f"{os.path.join(self.datadir, self._language)}.gui"
        referencePath = f"{os.path.join(self.datadir, self._language)}-1.txt"
        self._reference = self.readReference(referencePath)
        self._output = self.datadir + "/output"

    def testRecognizeRequest(self):
        hypothesis = self._recognizeAudio(self._audio, self._language)
        self.assertGreater(len(hypothesis), 1)
        self.evaluateHypothesis(self._reference, hypothesis)

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
        outs, errs = process.communicate(timeout=15)
        print("ERRORS:", errs)
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
            f"{self._output }/wer/test_{self._language}.pra"
        ), "analysis file does not exist"
        assert os.path.exists(
            f"{self._output }/wer/test_{self._language}.dtl"
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
            f"{self._output }/wer/test_{self._language}.pra"
        ), "analysis file does not exist"
        assert os.path.exists(
            f"{self._output }/wer/test_{self._language}.dtl"
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

    def testRecognizeRequestNoFormatted(self):
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
        self.assertGreater(len(hypothesis), 1)
        self.evaluateHypothesis(self._reference, hypothesis)
        self.ensureLowerCase(hypothesis)

    def ensureLowerCase(self, text):
        match = re.search("[A-Z]", text)
        self.assertEqual(match, None)

    def testRecognizeTimestampsExist(self):
        process = self.launchRecognitionProcess(self._audio, self._language)
        status = process.wait(timeout=900)
        self.checkStatus(status, process.stderr.read())
        message = self.__extractMessageAsAJsonObject(process.stdout.read())
        audioLength = parseSeconds(message["results"]["duration"])
        if message:
            stats = self.__calculateTimeStampsStats(
                message["results"]["alternatives"][0]["words"], audioLength
            )
            if self.__asrIsIssuingTimestamps(stats):
                self.assertGreater(stats.meanWordDuration, 0)
                self.assertTrue(stats.minWordDuration >= 0)  # no negative duration
                self.assertGreater(2, stats.maxWordDuration)  # extreme long words
                self.assertGreater(stats.numberOfWords, 5)  # test audios is long enough
                self.assertGreater(stats.speechTime, 0.20 * audioLength)
                self.assertGreater(stats.minSilenceDuration, 0)  # on negative durations
                self.assertGreater(stats.silenceTime, 0.25 * audioLength)
                self.assertGreater(
                    stats.maxSilenceDuration, 5
                )  # reasonable time for a test audio
                self.assertGreater(
                    stats.numberOfSilences, 1
                )  # initial and final silences at least

    @staticmethod
    def __extractMessageAsAJsonObject(text) -> Optional[object]:
        header = "Messages:"
        i = text.find(header)
        if i != -1:
            return json.loads(text[1 + len(header) + i :])
        return None

    @staticmethod
    def __asrIsIssuingTimestamps(stats: TimeStampsStatistics) -> bool:
        return stats.meanWordDuration != 0.0

    def __calculateTimeStampsStats(self, words, audioLength) -> TimeStampsStatistics:
        stats = TimeStampsStatistics()
        previousEnd = 0.0
        for word in words:
            stats.updateSilenceStats(parseSeconds(word["startTime"]) - previousEnd)
            stats.updateSpeechStats(
                parseSeconds(word["endTime"]) - parseSeconds(word["startTime"])
            )
        stats.updateSilenceStats(audioLength - previousEnd)
        return stats
