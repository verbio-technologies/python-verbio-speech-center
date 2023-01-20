import json
import os
import argparse


class ModelOutput:
    def __init__(self, model_accuracy_file, model_oov_file, model_intratest_folder):
        self.model_accuracy_file = model_accuracy_file
        self.model_oov_file = model_oov_file
        self.model_intratest_folder = model_intratest_folder
        self.accuracy = None
        self.oov = None
        self.domains = {}
        self.dialects = {}

    def load_model_accuracy_metric(self):
        with open(self.model_accuracy_file) as ac_file:

            for line in ac_file.readlines()[::-1]:
                if "Accuracy" in line:
                    self.accuracy = float(line.strip().split("\t")[1].split(" ")[1])
                    break
        if not self.accuracy:
            print("Could not load accuracy score from model output")

    def load_model_oov_metric(self):
        with open(self.model_oov_file) as oov_f:
            try:
                oov_info = json.load(oov_f)["score"]
                self.oov = float(oov_info.strip("%"))
            except:
                print("Could not load OOV score from model output")

    def load_intratest_data(self):
        for intratest_type in ["domains", "dialects"]:
            try:
                model_intratest_data = json.load(
                    open(
                        os.path.join(
                            self.model_intratest_folder,
                            f"{intratest_type}_intratest.json",
                        )
                    )
                )
                if intratest_type == "domains":
                    self.domains = model_intratest_data
                else:
                    self.dialects = model_intratest_data
            except FileNotFoundError:
                print(f"There is no data for {intratest_key}")

    def load_model_metrics(self):
        self.load_model_accuracy_metric()
        self.load_model_oov_metric()
        self.load_intratest_data()


class ExpectedOutput:
    def __init__(self, expected_output_file):
        self.expected_output_file = expected_output_file
        self.metrics_data = {}
        self.accuracy = None
        self.oov = None
        self.domains = {}
        self.dialects = {}
        self.domains_deviation = None
        self.dialects_deviation = None

    def load_expected_output_data(self):
        with open(self.expected_output_file) as ref_file:
            try:
                self.metrics_data = json.load(ref_file)
            except:
                print("Could not load reference metrics")

    def get_expected_global_metrics(self, language, test_type):
        try:
            self.accuracy = self.metrics_data[language][test_type]["accuracy"]
            self.oov = self.metrics_data[language][test_type]["oov"]
        except KeyError:
            print(f"Could not find {language} or {test_type} in reference")
            return None

    def get_expected_intratest_metrics(self, language, test_type):
        try:
            if "domains" in self.metrics_data[language][test_type]:
                self.domains = self.metrics_data[language][test_type]["domains"]
                self.domains_deviation = self.metrics_data[language][test_type][
                    "domains_typical_deviation"
                ]

            if "dialects" in self.metrics_data[language][test_type]:
                self.dialects = self.metrics_data[language][test_type]["dialects"]
                self.dialects_deviation = self.metrics_data[language][test_type][
                    "dialects_typical_deviation"
                ]
        except KeyError:
            print(f"Could not find {language} or {test_type} in reference")
            return None

    def load_expected_metrics(self, language, test_type):
        self.load_expected_output_data()
        self.get_expected_global_metrics(language, test_type)
        self.get_expected_intratest_metrics(language, test_type)


class Metrics:
    def __init__(
        self,
        model_results,
        expected_results,
        test_type,
        language,
    ):
        self.accuracy_tolerance = 1
        self.oov_tolerance = 0.25
        self.TEST_PASSED = True
        self.model_results = model_results
        self.expected_results = expected_results
        self.test_type = test_type
        self.language = language

    def compare_accuracy(self, model_result, expected_result, comparison_type):
        if model_result and expected_result:
            if abs(model_result - expected_result) <= self.accuracy_tolerance:
                print(
                    f"{comparison_type}: Model values and expected values match ({model_result})"
                )
            elif model_result > expected_result:
                print(
                    f"{comparison_type}: Model value ({model_result}) is higher than expected value ({expected_result})"
                )
            else:
                print(
                    f"{comparison_type}: Model value ({model_result}) is lower than expected value ({expected_result})"
                )
                print("Metric is below threshold: test wont pass")
                self.TEST_PASSED = False
        else:
            self.TEST_PASSED = False
            print(f"Test won't pass. Could not run accuracy check")

    def compare_oov(self):
        if self.model_results.oov and self.expected_results.oov:
            if (
                abs(self.model_results.oov - self.expected_results.oov)
                <= self.oov_tolerance
            ):
                print(
                    f"OOV: Model values and expected values match ({self.model_results.oov})"
                )
            elif self.model_results.oov > self.expected_results.oov:
                print(
                    f"OOV: Model value ({self.model_results.oov}) is higher than expected value ({self.expected_results.oov})"
                )
                self.TEST_PASSED = False
                print("Metric is above threshold: test wont pass")
            else:
                print(
                    f"OOV: Model value ({self.model_results.oov}) is lower than expected value ({self.expected_results.oov})"
                )
        else:
            self.TEST_PASSED = False
            print(f"Test won't pass. Could not run OOV check")

    def compare_deviation(self, intratest_type):
        if intratest_type == "domains":
            model_deviation = self.model_results.domains["Accuracy typical deviation"]
            reference_deviation = self.expected_results.domains_deviation
        else:
            model_deviation = self.model_results.dialects["Accuracy typical deviation"]
            reference_deviation = self.expected_results.dialects_deviation
        if model_deviation == reference_deviation:
            print(
                f"Deviation: intratest for {intratest_type}: Model values and expected values match ({model_deviation})"
            )
        else:
            print(
                f"Deviation: Intratest for {intratest_type}: Model values ({model_deviation}) and expected values ({reference_deviation}) do not match"
            )

    def compare_domains(self):
        for key in self.expected_results.domains:
            try:
                expected_accuracy = self.expected_results.domains[key]["accuracy"]
                model_accuracy = self.model_results.domains[key]
                self.compare_accuracy(model_accuracy, expected_accuracy, f" {key}")
            except KeyError:
                print(
                    f"Test won't pass. Could not find {key} in reference or intratest results file"
                )
                self.TEST_PASSED = False

    def compare_dialects(self):
        for key in self.expected_results.dialects:
            try:
                expected_accuracy = self.expected_results.dialects[key]["accuracy"]
                model_accuracy = self.model_results.dialects[key]
                self.compare_accuracy(model_accuracy, expected_accuracy, f" {key}")
            except KeyError:
                print(
                    f"Test won't pass. Could not find {key} in reference or intratest results file"
                )
                self.TEST_PASSED = False

    def compare_metrics(self):
        self.compare_accuracy(
            self.model_results.accuracy,
            self.expected_results.accuracy,
            "Global accuracy",
        )
        self.compare_oov()
        self.compare_deviation("dialects")
        self.compare_deviation("domains")
        self.compare_domains()
        self.compare_dialects()
        if self.TEST_PASSED == False:
            raise Exception("Test did not pass")


def main():
    parser = argparse.ArgumentParser(
        description="Test accuracy, OOV and intratest metrics"
    )
    parser.add_argument(
        "-w",
        "--model_accuracy",
        dest="model_accuracy",
        help="Tsv file containing model accuracy.",
    )
    parser.add_argument(
        "-o",
        "--model_oov",
        dest="model_oov",
        help="Json file containing model OOV metric.",
    )
    parser.add_argument(
        "-i",
        "--model_intratest_folder",
        dest="model_intratest_folder",
        help="Folder containing json files with model intratest metric.",
    )
    parser.add_argument(
        "-e",
        "--expected_metrics",
        dest="expected_metrics",
        help="Json file containing expected metrics.",
    )
    parser.add_argument(
        "-t", "--test_type", dest="test_type", help="Basic or upgraded test."
    )
    parser.add_argument("-l", "--language", dest="language", help="Language.")

    args = parser.parse_args()
    modelOutput = ModelOutput(
        args.model_accuracy, args.model_oov, args.model_intratest_folder
    )
    modelOutput.load_model_metrics()
    expectedOutput = ExpectedOutput(args.expected_metrics)
    expectedOutput.load_expected_metrics(args.language, args.test_type)
    metrics = Metrics(
        modelOutput,
        expectedOutput,
        args.test_type,
        args.language,
    )
    metrics.compare_metrics()


if __name__ == "__main__":
    main()
