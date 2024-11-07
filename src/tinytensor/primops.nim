# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import std / math

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# strided vector + helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# A view into a contiguous array with a stride
type
    StridedVector*[T; n: static int] = object
        data*: ptr UncheckedArray[T]  # Raw pointer to the first element
        stride*: int                   # Distance between consecutive elements

func `[]`*[T; n: static int](v: StridedVector[T, n], index: int): T =
    result = v.data[index * v.stride]

func `[]=`*[T; n: static int](v: StridedVector[T, n], index: int, value: T) =
    v.data[index * v.stride] = value

# proc init*[T](_: typedesc[StridedVector[T; n: static int]], shape: openArray[int]): Tensor[T] = newTensor[T](shape)

# ~~~~~~~~~~~~~~~~~~~~~
# arithmetic operations
# ~~~~~~~~~~~~~~~~~~~~~

func add*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] + b.data[i]

func addScalar*[T; n: static int](a: StridedVector[T, n], b: T, result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] + b

func subtract*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] - b.data[i]

func subtractScalar*[T; n: static int](a: StridedVector[T, n], b: T, result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] - b

func negate*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = -a.data[i]

func hadamard*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] * b.data[i]

func scale*[T; n: static int](a: StridedVector[T, n], b: T, result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] * b

# func `+`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] + B.data[i]

# func `+`*[T; shape: static TensorShape](A: Tensor[T, shape], a: T): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] + a

# func `+`*[T; shape: static TensorShape](a: T, A: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] + a

# func `-`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] - B.data[i]

# func `-`*[T; shape: static TensorShape](A: Tensor[T, shape], a: T): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] - a

# func `-`*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = -A.data[i]

# func hadamard*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] * B.data[i]

# func `*`*[T; shape: static TensorShape](A: Tensor[T, shape], a: T): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] * a

# func `*`*[T; shape: static TensorShape](a: T, A: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] * a

# ~~~~~~~~~~~~~~~~~~~~
# reduction operations
# ~~~~~~~~~~~~~~~~~~~~

func sum*[T; n: static int](a: StridedVector[T, n]): T =
    for i in 0..<n:
        result += a.data[i]

func mean*[T; n: static int](a: StridedVector[T, n]): T =
    result = a.sum() * (1.T / n.T)

func max*[T; n: static int](a: StridedVector[T, n]): T =
    result = a.data[0]
    for i in 1..<n:
        let val = a.data[i]
        if val > result:
            result = val

func min*[T; n: static int](a: StridedVector[T, n]): T =
    result = a.data[0]
    for i in 1..<n:
        let val = a.data[i]
        if val < result:
            result = val

func argmax*[T; n: static int](a: StridedVector[T, n]): int =
    result = 0
    var maxVal = a.data[0]
    for i in 1..<n:
        let val = a.data[i]
        if val > maxVal:
            maxVal = val
            result = i

func argmin*[T; n: static int](a: StridedVector[T, n]): int =
    result = 0
    var minVal = a.data[0]
    for i in 1..<n:
        let val = a.data[i]
        if val < minVal:
            minVal = val
            result = i


# func sum*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
#     for i in 0..<A.data.len:
#         result += A.data[i]

# func mean*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
#     result = A.sum() * (1 / A.data.len)

# func max*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
#     result = A.data[0]  # Initialize with first element
#     for i in 1..<A.data.len:
#         if A.data[i] > result:
#             result = A.data[i]

# func min*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
#     result = A.data[0]  # Initialize with first element
#     for i in 1..<A.data.len:
#         if A.data[i] < result:
#             result = A.data[i]

# func argmax*[T; shape: static TensorShape](A: Tensor[T, shape]): int =
#     result = 0
#     var maxVal = A.data[0]
#     for i in 1..<A.data.len:
#         if A.data[i] > maxVal:
#             maxVal = A.data[i]
#             result = i

# func argmin*[T; shape: static TensorShape](A: Tensor[T, shape]): int =
#     result = 0
#     var minVal = A.data[0]
#     for i in 1..<A.data.len:
#         if A.data[i] < minVal:
#             minVal = A.data[i]
#             result = i

# ~~~~~~~~~~~~~~~~~~~~~
# comparison operations
# ~~~~~~~~~~~~~~~~~~~~~

# TODO: make these return tensors of 1s and 0s (converted to type T) instead of bool?

func equal*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] == b.data[i]

func notEqual*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] != b.data[i]

func greater*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] > b.data[i]

func greaterEqual*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] >= b.data[i]

func less*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] < b.data[i]

func lessEqual*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i] <= b.data[i]

# func `==`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
#     result = initTensor[bool, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] == B.data[i]

# func `!=`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
#     result = initTensor[bool, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] != B.data[i]

# func `>`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
#     result = initTensor[bool, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] > B.data[i]

# func `>=`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
#     result = initTensor[bool, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] >= B.data[i]

# func `<`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
#     result = initTensor[bool, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] < B.data[i]

# func `<=`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[bool, shape] =
#     result = initTensor[bool, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] <= B.data[i]

# ~~~~~~~~~~~~~~~~~~~~~~~
# miscellaneous functions
# ~~~~~~~~~~~~~~~~~~~~~~~

func recip*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = 1.T / a.data[i]

func abs*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = abs(a.data[i])

func sqrt*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = sqrt(a.data[i])

func square*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        let val = a.data[i * a.stride]
        result.data[i] = val * val

func ln*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = ln(a.data[i])

func exp*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = exp(a.data[i])

func sin*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = sin(a.data[i])

func clamp*[T; n: static int](a: StridedVector[T, n], min_val, max_val: T, result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = clamp(a.data[i], min_val .. max_val)

# func recip*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = 1.T / A.data[i]

# func abs*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = abs(A.data[i])

# func sqrt*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = sqrt(A.data[i])

# func sq*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = A.data[i] * A.data[i]

# func ln*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = ln(A.data[i])

# func exp*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = exp(A.data[i])

# func sin*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = sin(A.data[i])

# func clamp*[T; shape: static TensorShape](A: Tensor[T, shape], min_val, max_val: T): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = clamp(A.data[i], min_val .. max_val) # TODO: avoid slice?

# ~~~~~~~~~~~~~~~~~~~~
# activation functions
# ~~~~~~~~~~~~~~~~~~~~

func identity*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result.data[i] = a.data[i]

func relu*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        let val = a.data[i]
        result.data[i] = if val > T(0): val else: T(0)

func sigmoid*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        let x = a.data[i]
        result.data[i] = 1.T / (1.T + exp(-x))

func tanh*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        let 
            x = a.data[i]
            expX = exp(x)
            expNegX = exp(-x)
        result.data[i] = (expX - expNegX) / (expX + expNegX)

# func identity*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
#     result = A

# func relu*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
#     result = initTensor[T, shape]()
#     for i in 0..<result.data.len:
#         result.data[i] = if A.data[i] > T(0): A.data[i] else: T(0)

# func sigmoid*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[T, shape] =
#     result = recip(T(1) + exp(-A))

# func tanh*[T; shape: static TensorShape](A: Tensor[T, shape]): Tensor[float, shape] =
#     let
#         negA = -A
#         expA = exp(A)
#         expNegA = exp(negA)
#         numerator = expA - expNegA
#         denominator = expA + expNegA
#     result = hadamard(numerator, recip(denominator))

# TODO:
# Element-wise Leaky ReLu (?)
# Element-wise Swish (?)
# Element-wise GELU (?)
# Element-wise ELU (?)