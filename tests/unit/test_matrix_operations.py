import unittest
import pytest
import numpy as np

from asr4_streaming.recognizer_v1.runtime.onnx import MatrixOperations


class TestMatrixOperation(unittest.TestCase):
    def setUp(self):
        self.window = 4
        self.overlap = 2
        self.audio = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

    def testSimple(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32)

        result = MatrixOperations(window=12, overlap=0).splitIntoOverlappingChunks(data)
        self.assertTrue((result[0] == data).all())

        result = MatrixOperations(window=12, overlap=1).splitIntoOverlappingChunks(data)
        self.assertEqual(result.shape, (1, 12))

        result = MatrixOperations(window=12, overlap=2).splitIntoOverlappingChunks(data)
        self.assertEqual(result.shape, (1, 12))

        result = MatrixOperations(window=5, overlap=1).splitIntoOverlappingChunks(data)
        self.assertEqual(result.shape, (3, 5))
        self.assertTrue((result[0] == [1, 2, 3, 4, 5]).all())
        self.assertTrue((result[1] == [5, 6, 7, 8, 9]).all())
        self.assertTrue((result[2] == [9, 10, 11, 12, 0]).all())

        result = MatrixOperations(window=4, overlap=1).splitIntoOverlappingChunks(data)
        self.assertEqual(result.shape, (4, 4))

        result = MatrixOperations(window=11, overlap=2).splitIntoOverlappingChunks(data)
        self.assertEqual(result.shape, (2, 11))

    def testExtremes(self):
        data = np.array([1], dtype=np.float32)

        result = MatrixOperations(window=1, overlap=0).splitIntoOverlappingChunks(data)
        self.assertEqual(result.shape, (1, 1))

        result = MatrixOperations(window=4, overlap=0).splitIntoOverlappingChunks(data)
        self.assertEqual(result.shape, (1, 4))

        data = np.array([1, 2, 3, 4], dtype=np.float32)
        result = MatrixOperations(window=4, overlap=0).splitIntoOverlappingChunks(data)
        self.assertEqual(result.shape, (1, 4))

        result = MatrixOperations(window=1, overlap=0).splitIntoOverlappingChunks(data)
        self.assertEqual(result.shape, (4, 1))

        result = MatrixOperations(window=2, overlap=0).splitIntoOverlappingChunks(data)
        self.assertEqual(result.shape, (2, 2))

        with pytest.raises(AssertionError):
            MatrixOperations(window=2, overlap=3).splitIntoOverlappingChunks(data)

    def testSplitIntoOverlappingChunks(self):
        result = MatrixOperations(
            window=self.window, overlap=self.overlap
        ).splitIntoOverlappingChunks(self.audio)

        # Expected output
        expected = np.array(
            [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]], dtype=np.float32
        )

        # Assert that the result matches the expected output
        np.testing.assert_array_equal(result, expected)

    def testSplitIntoOverlappingChunksFill(self):
        result = MatrixOperations(window=3, overlap=1).splitIntoOverlappingChunks(
            self.audio
        )

        # Expected output
        expected = np.array(
            [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 0]], dtype=np.float32
        )

        # Assert that the result matches the expected output
        np.testing.assert_array_equal(result, expected)

    def testSplitIntoOverlappingChunksZeroOverLap(self):
        # Test with zero overlap
        result = MatrixOperations(
            window=self.window, overlap=0
        ).splitIntoOverlappingChunks(self.audio)

        # Expected output
        expected = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 0, 0]], dtype=np.float32
        )

        # Assert that the result matches the expected output
        np.testing.assert_array_equal(result, expected)

    def testSplitIntoOverlappingChunksLargerInput(self):
        # Test with a larger input array
        audio = np.arange(100, dtype=np.float32)
        window = 10
        overlap = 5

        result = MatrixOperations(
            window=window, overlap=overlap
        ).splitIntoOverlappingChunks(audio)

        # Calculate expected output
        expected = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                [25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                [35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                [55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                [65, 66, 67, 68, 69, 70, 71, 72, 73, 74],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                [75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                [85, 86, 87, 88, 89, 90, 91, 92, 93, 94],
                [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            ],
            dtype=np.float32,
        )

        # Assert that the result matches the expected output
        np.testing.assert_array_equal(result, expected)

    def testSplitIntoOverlappingChunksSmallerInput(self):
        # Test with a smaller input array that is not divisible by window size
        audio = np.arange(5, dtype=np.float32)
        window = 10
        overlap = 2

        result = MatrixOperations(
            window=window, overlap=overlap
        ).splitIntoOverlappingChunks(audio)

        expected = np.array([[0, 1, 2, 3, 4, 0, 0, 0, 0, 0]], dtype="float32")

        # Assert that the result matches the expected output
        np.testing.assert_array_equal(result, expected)
