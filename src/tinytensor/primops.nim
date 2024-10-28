# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import std / math
import tensors

# ~~~~~~~~~~~~~~~~~~~~~
# arithmetic operations
# ~~~~~~~~~~~~~~~~~~~~~

func `+`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] + B.data[i]

func `+`*[T; shape: static TensorShape](A: Tensor[T, shape], a: T): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] + a

func `+`*[T; shape: static TensorShape](a: T, A: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] + a

func `-`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] - B.data[i]

func `-`*[T; shape: static TensorShape](A: Tensor[T, shape], a: T): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] - a

func `-`*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = -A.data[i]

func `*`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] * B.data[i]

func `*`*[T; shape: static TensorShape](A: Tensor[T, shape], a: T): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] * a

func `*`*[T; shape: static TensorShape](a: T, A: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] * a

# ~~~~~~~~~~~~~~~~~~~~
# reduction operations
# ~~~~~~~~~~~~~~~~~~~~

func sum*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
    for i in 0..<A.data.len:
        result += A.data[i]

func mean*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
    result = A.sum() * (1 / A.data.len)

func max*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
    result = A.data[0]  # Initialize with first element
    for i in 1..<A.data.len:
        if A.data[i] > result:
            result = A.data[i]

func min*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
    result = A.data[0]  # Initialize with first element
    for i in 1..<A.data.len:
        if A.data[i] < result:
            result = A.data[i]

func argmax*[T; shape: static TensorShape](A: Tensor[T, shape]): int =
    result = 0
    var maxVal = A.data[0]
    for i in 1..<A.data.len:
        if A.data[i] > maxVal:
            maxVal = A.data[i]
            result = i

func argmin*[T; shape: static TensorShape](A: Tensor[T, shape]): int =
    result = 0
    var minVal = A.data[0]
    for i in 1..<A.data.len:
        if A.data[i] < minVal:
            minVal = A.data[i]
            result = i

# ~~~~~~~~~~~~~~~~~~~~~
# comparison operations
# ~~~~~~~~~~~~~~~~~~~~~

# TODO: make these return tensors of 1s and 0s (converted to type T) instead of bool?

func `==`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
    result = initTensor[bool, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] == B.data[i]

func `!=`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
    result = initTensor[bool, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] != B.data[i]

func `>`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
    result = initTensor[bool, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] > B.data[i]

func `>=`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
    result = initTensor[bool, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] >= B.data[i]

func `<`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
    result = initTensor[bool, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] < B.data[i]

func `<=`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
    result = initTensor[bool, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] <= B.data[i]

# ~~~~~~~~~~~~~~~~~~~~~~~
# miscellaneous functions
# ~~~~~~~~~~~~~~~~~~~~~~~

func recip*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = 1.T / A.data[i]

func abs*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = abs(A.data[i])

func sqrt*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = sqrt(A.data[i])

func sq*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] * A.data[i]

func ln*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = ln(A.data[i])

func exp*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = exp(A.data[i])

func sin*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = sin(A.data[i])

func clamp*[T; shape: static TensorShape](A: Tensor[T, shape], min_val, max_val: T): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = clamp(A.data[i], min_val .. max_val) # TODO: avoid slice?

# ~~~~~~~~~~~~~~~~~~~~
# activation functions
# ~~~~~~~~~~~~~~~~~~~~

func identity*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
    result = A

func relu*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = if A.data[i] > T(0): A.data[i] else: T(0)

func sigmoid*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
    result = recip(T(1) + exp(-A))

func tanh*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[float, shape] =
    let
        negA = -A
        expA = exp(A)
        expNegA = exp(negA)
        numerator = expA - expNegA
        denominator = expA + expNegA
    result = numerator * recip(denominator)

# TODO:
# Element-wise Leaky ReLu (?)
# Element-wise Swish (?)
# Element-wise GELU (?)
# Element-wise ELU (?)