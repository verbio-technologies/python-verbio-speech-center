import unittest
import pytest
import logging
import numpy as np
import numpy.typing as npt

from asr4.recognizer_v1.runtime.onnx import MatrixOperations



class TestMatrixOperation(unittest.TestCase):
    def testSimple(self):
        m = MatrixOperations()

        data = np.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype=np.float32)

        result = m.splitToOverlappingChunks(data, 12, 0)
        self.assertTrue( (result[0]==data).all() )

        result = m.splitToOverlappingChunks(data, 12, 1)
        self.assertEqual( result.shape, (2,12) )

        result = m.splitToOverlappingChunks(data, 12, 2)
        self.assertEqual( result.shape, (2,12) )

        result = m.splitToOverlappingChunks(data, 5, 1)
        self.assertEqual( result.shape, (5,5) )
        self.assertTrue( (result[0] == [0,1,2,3,4]).all() )
        self.assertTrue( (result[1] == [3,4,5,6,7]).all() )
        self.assertTrue( (result[2] == [6,7,8,9,10]).all() )
        self.assertTrue( (result[3] == [9,10,11,12,0]).all() )
        self.assertTrue( (result[4] == [12,0,0,0,0]).all() )
        
        result = m.splitToOverlappingChunks(data, 4, 1)
        self.assertEqual( result.shape, (7,4) )

        result = m.splitToOverlappingChunks(data, 11, 2)
        self.assertEqual( result.shape, (2,11) )


    def testExtremes(self):
        m = MatrixOperations()

        data = np.array([1], dtype=np.float32)

        result = m.splitToOverlappingChunks(data, 1, 0)
        self.assertEqual( result.shape, (1,1) )

        result = m.splitToOverlappingChunks(data, 4, 0)
        self.assertEqual( result.shape, (1,4) )

        data = np.array([1,2,3,4], dtype=np.float32)

        result = m.splitToOverlappingChunks(data, 4, 0)
        self.assertEqual( result.shape, (1,4) )

        result = m.splitToOverlappingChunks(data, 1, 0)
        self.assertEqual( result.shape, (4,1) )
        
        result = m.splitToOverlappingChunks(data, 2, 0)
        self.assertEqual( result.shape, (2,2) )

        with pytest.raises(AssertionError):
            m.splitToOverlappingChunks(data, 3, 2)

