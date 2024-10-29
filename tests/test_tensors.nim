# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import tinytensor
import math

suite "Primitive Operations Tests":
    const
        # Define test shapes
        shape2D: TensorShape[2] = [2, 2]
        shape1D: TensorShape[1] = [4]
        epsilon: float = 1e-6  # For float comparisons

    test "Arithmetic Operations":
        # Initialize test tensors
        var 
            A = initTensor[float, shape2D]()
            B = initTensor[float, shape2D]()
        
        # Set test values
        A[0, 0] = 1.0; A[0, 1] = 2.0
        A[1, 0] = 3.0; A[1, 1] = 4.0
        
        B[0, 0] = 0.5; B[0, 1] = 1.5
        B[1, 0] = 2.5; B[1, 1] = 3.5

        # Test addition
        let C = A + B
        assert abs(C[0, 0] - 1.5) < epsilon
        assert abs(C[0, 1] - 3.5) < epsilon
        assert abs(C[1, 0] - 5.5) < epsilon
        assert abs(C[1, 1] - 7.5) < epsilon

        # Test scalar addition
        let D = A + 1.0
        assert abs(D[0, 0] - 2.0) < epsilon
        assert abs(D[0, 1] - 3.0) < epsilon
        assert abs(D[1, 0] - 4.0) < epsilon
        assert abs(D[1, 1] - 5.0) < epsilon

        # Test subtraction
        let E = A - B
        assert abs(E[0, 0] - 0.5) < epsilon
        assert abs(E[0, 1] - 0.5) < epsilon
        assert abs(E[1, 0] - 0.5) < epsilon
        assert abs(E[1, 1] - 0.5) < epsilon

        # Test scalar subtraction
        let F = A - 1.0
        assert abs(F[0, 0] - 0.0) < epsilon
        assert abs(F[0, 1] - 1.0) < epsilon
        assert abs(F[1, 0] - 2.0) < epsilon
        assert abs(F[1, 1] - 3.0) < epsilon

        # Test multiplication
        let G = hadamard(A, B)
        assert abs(G[0, 0] - 0.5) < epsilon
        assert abs(G[0, 1] - 3.0) < epsilon
        assert abs(G[1, 0] - 7.5) < epsilon
        assert abs(G[1, 1] - 14.0) < epsilon

        # Test scalar multiplication
        let H = A * 2.0
        assert abs(H[0, 0] - 2.0) < epsilon
        assert abs(H[0, 1] - 4.0) < epsilon
        assert abs(H[1, 0] - 6.0) < epsilon
        assert abs(H[1, 1] - 8.0) < epsilon

    test "Reduction Operations":
        var A = initTensor[float, shape2D]()
        A[0, 0] = 1.0; A[0, 1] = 2.0
        A[1, 0] = 3.0; A[1, 1] = 4.0

        # Test sum
        assert abs(sum(A) - 10.0) < epsilon

        # Test mean
        assert abs(mean(A) - 2.5) < epsilon

        # Test max
        assert abs(max(A) - 4.0) < epsilon

        # Test min
        assert abs(min(A) - 1.0) < epsilon

        # Test argmax
        assert argmax(A) == 3  # 4.0 is at index 3

        # Test argmin
        assert argmin(A) == 0  # 1.0 is at index 0

    test "Comparison Operations":
        var 
            A = initTensor[float, shape2D]()
            B = initTensor[float, shape2D]()
        
        A[0, 0] = 1.0; A[0, 1] = 2.0
        A[1, 0] = 3.0; A[1, 1] = 4.0
        
        B[0, 0] = 1.0; B[0, 1] = 2.0
        B[1, 0] = 3.0; B[1, 1] = 3.0

        # Test equality
        let eq = A == B
        assert eq[0, 0] == true
        assert eq[0, 1] == true
        assert eq[1, 0] == true
        assert eq[1, 1] == false

        # Test inequality
        let neq = A != B
        assert neq[0, 0] == false
        assert neq[0, 1] == false
        assert neq[1, 0] == false
        assert neq[1, 1] == true

        # Test greater than
        let gt = A > B
        assert gt[0, 0] == false
        assert gt[0, 1] == false
        assert gt[1, 0] == false
        assert gt[1, 1] == true

    test "Miscellaneous Functions":
        var A = initTensor[float, shape2D]()
        A[0, 0] = 1.0; A[0, 1] = 2.0
        A[1, 0] = 3.0; A[1, 1] = 4.0

        # Test reciprocal
        let rec = recip(A)
        assert abs(rec[0, 0] - 1.0) < epsilon
        assert abs(rec[0, 1] - 0.5) < epsilon
        assert abs(rec[1, 0] - 0.3333333333333333) < epsilon
        assert abs(rec[1, 1] - 0.25) < epsilon

        # Test absolute value
        var B = initTensor[float, shape2D]()
        B[0, 0] = -1.0; B[0, 1] = 2.0
        B[1, 0] = -3.0; B[1, 1] = 4.0
        
        let absB = abs(B)
        assert abs(absB[0, 0] - 1.0) < epsilon
        assert abs(absB[0, 1] - 2.0) < epsilon
        assert abs(absB[1, 0] - 3.0) < epsilon
        assert abs(absB[1, 1] - 4.0) < epsilon

        # Test square
        let sq = sq(A)
        assert abs(sq[0, 0] - 1.0) < epsilon
        assert abs(sq[0, 1] - 4.0) < epsilon
        assert abs(sq[1, 0] - 9.0) < epsilon
        assert abs(sq[1, 1] - 16.0) < epsilon

        # Test square root
        let sqrtA = sqrt(A)
        assert abs(sqrtA[0, 0] - 1.0) < epsilon
        assert abs(sqrtA[0, 1] - 1.4142135623730951) < epsilon
        assert abs(sqrtA[1, 0] - 1.7320508075688772) < epsilon
        assert abs(sqrtA[1, 1] - 2.0) < epsilon

    test "Activation Functions":
        var A = initTensor[float, shape2D]()
        A[0, 0] = -1.0; A[0, 1] = 0.0
        A[1, 0] = 1.0; A[1, 1] = 2.0

        # Test ReLU
        let reluA = relu(A)
        assert abs(reluA[0, 0] - 0.0) < epsilon
        assert abs(reluA[0, 1] - 0.0) < epsilon
        assert abs(reluA[1, 0] - 1.0) < epsilon
        assert abs(reluA[1, 1] - 2.0) < epsilon

        # Test sigmoid
        let sigA = sigmoid(A)
        assert abs(sigA[0, 0] - 0.2689414213699951) < epsilon
        assert abs(sigA[0, 1] - 0.5) < epsilon
        assert abs(sigA[1, 0] - 0.7310585786300049) < epsilon
        assert abs(sigA[1, 1] - 0.8807970779778823) < epsilon

        # Test tanh
        let tanhA = tanh(A)
        assert abs(tanhA[0, 0] - (-0.7615941559557649)) < epsilon
        assert abs(tanhA[0, 1] - 0.0) < epsilon
        assert abs(tanhA[1, 0] - 0.7615941559557649) < epsilon
        assert abs(tanhA[1, 1] - 0.9640275800758169) < epsilon