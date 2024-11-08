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

# TODO: dot product?

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
        result.data[i] = clamp(a.data[i], min_val .. max_val) # TODO: avoid slice?

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

# TODO:
# Element-wise Leaky ReLu (?)
# Element-wise Swish (?)
# Element-wise GELU (?)
# Element-wise ELU (?)