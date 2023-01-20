import unittest
import argparse
import os
import json
import pytest
from pathlib import Path
from tests.e2e.metrics import Metrics, ModelOutput, ExpectedOutput


class MockModelOutput(ModelOutput):
    def __init__(self):
        self.data_parent = Path(__file__).parent
        self.model_accuracy_file = os.path.join(
            self.data_parent, "data/test_es_results.tsv"
        )
        self.model_oov_file = os.path.join(self.data_parent, "data/test_es_oov.json")
        self.model_intratest_folder = os.path.join(
            self.data_parent, "data/test_es_intratest/"
        )
        self.golden_domains = {
            "multidomain-real_estate-hh-servihabitat": 55.9,
            "multidomain-technology_support-hh-bq_tech_support": 78.23,
            "multidomain-telco-hm-telefónica": 86.44,
            "multidomain-telco-hm-tmobile": 72.22,
            "Accuracy typical deviation": 11.19,
        }
        self.golden_dialects = {
            "es-co": 86.44,
            "es-es": 63.34,
            "es-us": 72.22,
            "Accuracy typical deviation": 9.51,
        }
        super().__init__(
            self.model_accuracy_file, self.model_oov_file, self.model_intratest_folder
        )

    def set_empty_files(self):
        empty_file = os.path.join(self.data_parent, "data/empty_file.txt")
        self.model_accuracy_file = empty_file
        self.model_oov_file = empty_file


class MockExpectedOutput(ExpectedOutput):
    def __init__(self):
        self.data_parent = Path(__file__).parent
        self.expected_output_file = os.path.join(
            self.data_parent, "data/expected_metrics.json"
        )
        super().__init__(self.expected_output_file)

    def set_empty_file(self):
        empty_file = os.path.join(self.data_parent, "data/empty_file.txt")
        self.expected_output_file = empty_file


class TestLoadModelOutput(unittest.TestCase):
    def testLoadFiles(self):
        mockModelOutput = MockModelOutput()
        mockModelOutput.load_model_metrics()
        self.assertEqual(mockModelOutput.accuracy, 62.15)
        self.assertEqual(mockModelOutput.oov, 5.59)
        self.assertEqual(mockModelOutput.domains, mockModelOutput.golden_domains)
        self.assertEqual(mockModelOutput.dialects, mockModelOutput.golden_dialects)

    def testEmptyFiles(self):
        mockModelOutput = MockModelOutput()
        mockModelOutput.set_empty_files()
        mockModelOutput.load_model_metrics()
        self.assertEqual(mockModelOutput.accuracy, None)
        self.assertEqual(mockModelOutput.oov, None)


class TestLoadExpectedOutput(unittest.TestCase):
    def testLoadFiles(self):
        mockExpectedOutput = MockExpectedOutput()
        mockExpectedOutput.load_expected_metrics("es", "basic")
        self.assertEqual(mockExpectedOutput.accuracy, 61.07)
        self.assertEqual(mockExpectedOutput.oov, 6.37)
        self.assertEqual(len(mockExpectedOutput.domains), 16)
        self.assertEqual(len(mockExpectedOutput.dialects), 3)
        self.assertEqual(mockExpectedOutput.domains_deviation, 18.09)
        self.assertEqual(mockExpectedOutput.dialects_deviation, 5.15)

    def testEmptyFiles(self):
        mockExpectedOutput = MockExpectedOutput()
        mockExpectedOutput.set_empty_file()
        mockExpectedOutput.load_expected_metrics("es", "basic")
        self.assertEqual(mockExpectedOutput.metrics_data, {})


class TestCompareMetrics(unittest.TestCase):
    def setUp(self):
        self.mockExpectedOutput = MockExpectedOutput()
        self.mockExpectedOutput.load_expected_metrics("es", "basic")
        self.mockModelOutput = MockModelOutput()
        self.mockModelOutput.load_model_metrics()

    def testCompareSameAccuracy(self):
        self.mockModelOutput.accuracy = 20
        self.mockExpectedOutput.accuracy = 20
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_accuracy(20, 20, "Test type")
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareHigherModelAccuracy(self):
        self.mockModelOutput.accuracy = 40
        self.mockExpectedOutput.accuracy = 20
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_accuracy(40, 20, "Accuracy")
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareHigherReferenceAccuracy(self):
        self.mockModelOutput.accuracy = 20
        self.mockExpectedOutput.accuracy = 40
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_accuracy(20, 40, "Accuracy")
        self.assertEqual(metrics.TEST_PASSED, False)

    def testCompareHigherReferenceWithinThresholdAccuracy(self):
        self.mockModelOutput.accuracy = 39
        self.mockExpectedOutput.accuracy = 40
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_accuracy(39, 40, "Accuracy")
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareHigherModelOOV(self):
        self.mockModelOutput.oov = 5
        self.mockExpectedOutput.oov = 4
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_oov()
        self.assertEqual(metrics.TEST_PASSED, False)

    def testCompareHigherModelWithinThresholdOOV(self):
        self.mockModelOutput.oov = 5
        self.mockExpectedOutput.oov = 4.75
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_oov()
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareLowerModelOOV(self):
        self.mockModelOutput.oov = 4
        self.mockExpectedOutput.oov = 5
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_oov()
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareHigherModelDeviation(self):
        self.mockModelOutput.domains = {"Accuracy typical deviation": 9.51}
        self.mockExpectedOutput.domains_deviation = 5
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_deviation("intratest domain name")
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareLowerModelDeviation(self):
        self.mockModelOutput.domains = {"Accuracy typical deviation": 0.51}
        self.mockExpectedOutput.domains_deviation = 5
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_deviation("intratest domain name")
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareDomainsMissingDomains(self):
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_domains()
        self.assertEqual(metrics.TEST_PASSED, False)

    def testCompareDomainsPass(self):
        self.mockModelOutput.domains = {"domain1": 45.12, "domain2": 35.2}
        self.mockExpectedOutput.domains = {
            "domain1": {"accuracy": 5.12},
            "domain2": {"accuracy": 35.2},
        }
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_domains()
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareDomainsFail(self):
        self.mockModelOutput.domains = {"domain1": 5.12, "domain2": 35.2}
        self.mockExpectedOutput.domains = {
            "domain1": {"accuracy": 45.12},
            "domain2": {"accuracy": 35.2},
        }
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_domains()
        self.assertEqual(metrics.TEST_PASSED, False)

    def testCompareDomainsMissingDialects(self):
        self.mockModelOutput.dialects = {"es-co": 86.44}
        self.mockExpectedOutput.dialects = {
            "es-co": {"accuracy": 48.12},
            "es-es": {"accuracy": 53.96},
            "es-us": {"accuracy": 53.98},
        }
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_dialects()
        self.assertEqual(metrics.TEST_PASSED, False)

    def testCompareDialectsPass(self):
        self.mockModelOutput.dialects = {"es-co": 86.44, "es-es": 63.34, "es-us": 72.22}
        self.mockExpectedOutput.dialects = {
            "es-co": {"accuracy": 48.12},
            "es-es": {"accuracy": 53.96},
            "es-us": {"accuracy": 53.98},
        }
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_dialects()
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareDialectsFail(self):
        self.mockModelOutput.dialects = {"es-co": 86.44, "es-es": 63.34, "es-us": 72.22}
        self.mockExpectedOutput.dialects = {
            "es-co": {"accuracy": 48.12},
            "es-es": {"accuracy": 53.96},
            "es-us": {"accuracy": 100},
        }
        metrics = Metrics(self.mockModelOutput, self.mockExpectedOutput, "basic", "es")
        metrics.compare_dialects()
        self.assertEqual(metrics.TEST_PASSED, False)
