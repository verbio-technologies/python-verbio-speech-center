import unittest
import argparse
import os
import json
from pathlib import Path
from tests.e2e.metrics import Metrics


class MockMetrics(Metrics):
    def __init__(self):
        self.data_parent = Path(__file__).parent
        obtained_accuracy = os.path.join(self.data_parent, "data/test_es_results.tsv")
        obtained_oov = os.path.join(self.data_parent, "data/test_es_oov.json")
        obtained_intratest_folder = os.path.join(
            self.data_parent, "data/test_es_intratest/"
        )
        expected_metrics = os.path.join(self.data_parent, "data/expected_metrics.json")
        test_type = "basic"
        language = "es"
        super().__init__(
            obtained_accuracy,
            obtained_oov,
            obtained_intratest_folder,
            expected_metrics,
            test_type,
            language,
        )

    def set_empty_accuracy_file(self):
        empty_obtained_accuracy = os.path.join(self.data_parent, "data/empty_file.txt")
        self.obtained_accuracy = empty_obtained_accuracy


class TestCompareMetrics(unittest.TestCase):
    def testCompareSameAccuracy(self):
        metrics = MockMetrics()
        metrics.compare_metrics(30, 30, "Accuracy")
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareHigherModelAccuracy(self):
        metrics = MockMetrics()
        metrics.compare_metrics(40, 20, "Accuracy")
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareHigherReferenceAccuracy(self):
        metrics = MockMetrics()
        metrics.compare_metrics(10, 50, "Accuracy")
        self.assertEqual(metrics.TEST_PASSED, False)

    def testCompareHigherReferenceWithinThresholdAccuracy(self):
        metrics = MockMetrics()
        metrics.compare_metrics(45, 46, "Accuracy")
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareHigherModelOOV(self):
        metrics = MockMetrics()
        metrics.compare_metrics("+3", 2.5, "OOV")
        self.assertEqual(metrics.TEST_PASSED, False)

    def testCompareHigherModelWithinThresholdOOV(self):
        metrics = MockMetrics()
        metrics.compare_metrics("+3", 2.75, "OOV")
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareLowerModelOOV(self):
        metrics = MockMetrics()
        metrics.compare_metrics("+2", 2.75, "OOV")
        self.assertEqual(metrics.TEST_PASSED, True)

    def testCompareHigherReferenceIntratest(self):
        metrics = MockMetrics()
        metrics.compare_metrics(45, 46, "Intratest")
        self.assertEqual(metrics.TEST_PASSED, True)


class TestGetMetrics(unittest.TestCase):
    def testGetAccuracyModel(self):
        metrics = MockMetrics()
        accuracy = metrics.get_accuracy_obtained_metric()
        self.assertEqual(accuracy, "62.15")

    def testGetOOVModel(self):
        metrics = MockMetrics()
        OOV = metrics.get_oov_obtained_metric()
        self.assertEqual(OOV, "+5.59")

    def testGetExpectedMetric(self):
        metrics = MockMetrics()
        expected_oov = metrics.get_expected_metric("oov")
        expected_accuracy = metrics.get_expected_metric("accuracy")
        random_metric = metrics.get_expected_metric("random")
        self.assertEqual(expected_oov, 6.37)
        self.assertEqual(expected_accuracy, 61.07)
        self.assertEqual(random_metric, None)

    def testMakeComparisonTest(self):
        metrics = MockMetrics()
        metrics.make_accuracy_comparison()
        metrics.make_oov_comparison()
        self.assertEqual(metrics.TEST_PASSED, True)

    def testMakeComparisonTestEmptyAccuracyFile(self):
        metrics = MockMetrics()
        metrics.set_empty_accuracy_file()
        metrics.make_accuracy_comparison()
        metrics.make_oov_comparison()
        self.assertEqual(metrics.TEST_PASSED, False)

    def testMakeIntratestComparison(self):
        metrics = MockMetrics()
        reference_data = json.load(open(metrics.expected_metrics))[metrics.language][
            metrics.test_type
        ]
        metrics.make_intratest_comparison(reference_data, "dialects")
        self.assertEqual(metrics.TEST_PASSED, True)

    def testMakeIntratestComparisonLowModelAccuracy(self):
        metrics = MockMetrics()
        reference_data = {
            "domains_typical_deviation": 5.15,
            "domains": {"multidomain-real_estate-hh-servihabitat": {"accuracy": 100}},
        }
        metrics.make_intratest_comparison(reference_data, "domains")
        self.assertEqual(metrics.TEST_PASSED, False)

    def testMakeIntratestComparisonMissingData(self):
        metrics = MockMetrics()
        reference_data = {
            "domains_typical_deviation": 5.15,
            "domains": {"domain-only-in-reference": {"accuracy": 0}},
        }
        metrics.make_intratest_comparison(reference_data, "domains")
        self.assertEqual(metrics.TEST_PASSED, False)
