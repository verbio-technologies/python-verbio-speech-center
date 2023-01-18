import json
import os
import argparse


class Metrics:
	def __init__(
		self,
		obtained_accuracy,
		obtained_oov,
		obtained_intratest_folder,
		expected_metrics,
		test_type,
		language,
	):
		self.accuracy_interval = 1
		self.oov_interval = 0.25
		self.TEST_PASSED = True
		self.obtained_accuracy = obtained_accuracy
		self.obtained_oov = obtained_oov
		self.obtained_intratest_folder = obtained_intratest_folder
		self.expected_metrics = expected_metrics
		self.test_type = test_type
		self.language = language

	def compare_metrics(self, obtained_metric, expected_metric, comparison_type):
		obtained_metric = float(obtained_metric)
		if comparison_type == "OOV":
			interval = self.oov_interval
		else:
			interval = self.accuracy_interval
		if abs(obtained_metric - expected_metric) <= interval:
			print(
				f"{comparison_type}: Obtained values and expected values match ({obtained_metric})"
			)
		elif obtained_metric > expected_metric:
			print(
				f"{comparison_type}: Obtained value ({obtained_metric}) is higher than expected value ({expected_metric})"
			)
			if comparison_type == "OOV":
				self.TEST_PASSED = False
				print("Metric is above threshold: test wont pass")
		else:
			print(
				f"{comparison_type}: Obtained value ({obtained_metric}) is lower than expected value ({expected_metric})"
			)
			if comparison_type != "OOV":
				self.TEST_PASSED = False
				print("Metric is below threshold: test wont pass")


	def get_accuracy_obtained_metric(self):
		with open(self.obtained_accuracy) as ac_file:
			for line in ac_file.readlines()[::-1]:
				if "Accuracy" in line:
					return line.strip().split("\t")[1].split(" ")[1]

	def get_oov_obtained_metric(self):
		with open(self.obtained_oov) as oov_f:
			oov_info = json.load(oov_f)["score"]
			return oov_info.strip("%")

	def get_expected_metric(self, metric):
		with open(self.expected_metrics) as ref_file:
			expected_metrics_file = json.load(ref_file)
			try:
				expected_metric = expected_metrics_file[self.language][self.test_type][metric]
				return expected_metric
			except KeyError:
				self.TEST_PASSED = False
				print(f"Test won't pass. Could not find {self.language}, {self.test_type} or {metric} in reference")

	def make_accuracy_comparison(self):
		obtained_accuracy = self.get_accuracy_obtained_metric()
		expected_accuracy = self.get_expected_metric("accuracy")
		if obtained_accuracy and expected_accuracy:
			self.compare_metrics(obtained_accuracy, expected_accuracy, "Accuracy")
		else:
			self.TEST_PASSED = False
			print(f"Test won't pass. Could not run accuracy check")       

	def make_oov_comparison(self):
		obtained_oov = self.get_oov_obtained_metric()
		expected_oov = self.get_expected_metric("oov")
		if obtained_oov and expected_oov:
			self.compare_metrics(obtained_oov, expected_oov, "OOV")
		else:
			self.TEST_PASSED = False
			print(f"Test won't pass. Could not run OOV check")

	def make_intratest_comparison(self, reference_data, intratest_type):
		obtained_data_contents = json.load(
			open(
				os.path.join(
					self.obtained_intratest_folder, f"{intratest_type}_intratest.json"
				)
			)
		)
		obtained_deviation = obtained_data_contents["Accuracy typical deviation"]
		reference_deviation = reference_data[f"{intratest_type}_typical_deviation"]
		if obtained_deviation == reference_deviation:
			print(
				f"Deviation: intratest for {intratest_type}: Obtained values and expected values match ({obtained_deviation})"
			)
		else:
			print(
				f"Deviation: Intratest for {intratest_type}: Obtained values ({obtained_deviation}) and expected values ({reference_deviation}) do not match"
			)

		for key in reference_data[intratest_type]:
			try:
				expected_accuracy = reference_data[intratest_type][key]["accuracy"]
				obtained_accuracy = obtained_data_contents[key]
				self.compare_metrics(obtained_accuracy, expected_accuracy, intratest_type + f" {key}")
			except KeyError:
				print(
					f"Test won't pass. Could not find {key} in reference or intratest results file)"
				)
				self.TEST_PASSED = False

	def launch_metric_test(self):
		self.make_accuracy_comparison()
		self.make_oov_comparison()
		reference_data = json.load(open(self.expected_metrics))[self.language][
			self.test_type
		]
		if "domains" in reference_data:

			self.make_intratest_comparison(reference_data, "domains")
		if "dialects" in reference_data:
			self.make_intratest_comparison(reference_data, "dialects")

		if self.TEST_PASSED == False:
			raise Exception("Test did not pass")


def main():
	parser = argparse.ArgumentParser(description="Test accuracy, OOV and intratest metrics")
	parser.add_argument(
		"-w",
		"--obtained_accuracy",
		dest="obtained_accuracy",
		help="Tsv file containing obtained accuracy.",
	)
	parser.add_argument(
		"-o",
		"--obtained_oov",
		dest="obtained_oov",
		help="Json file containing obtained OOV metric.",
	)
	parser.add_argument(
		"-i",
		"--obtained_intratest_folder",
		dest="obtained_intratest_folder",
		help="Folder containing json files with obtained intratest metric.",
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
	metrics = Metrics(
		args.obtained_accuracy,
		args.obtained_oov,
		args.obtained_intratest_folder,
		args.expected_metrics,
		args.test_type,
		args.language,
	)
	metrics.launch_metric_test()


if __name__ == "__main__":
	main()
