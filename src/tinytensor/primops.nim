# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensors

# ~~~~~~~~~~~~~~~~~~~~~
# arithmetic operations
# ~~~~~~~~~~~~~~~~~~~~~

func `+`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] + B.data[i]

func `-`*[T; shape: static TensorShape](A, B: Tensor[T, shape]): Tensor[T, shape] =
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = A.data[i] - B.data[i]

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
    result = A.sum() / A.data.len()

func max*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
    # TODO
    result = A.sum() / A.data.len()

func min*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
    # TODO
    result = A.sum() / A.data.len()

func argmax*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
    # TODO
    result = A.sum() / A.data.len()

func argmin*[T; shape: static TensorShape](A: Tensor[T, shape]): T =
    # TODO
    result = A.sum() / A.data.len()

# ~~~~~~~~~~~~~~~~~~
# logical operations
# ~~~~~~~~~~~~~~~~~~

# TODO:
# 1. Element-wise Equal
# 2. Element-wise Not Equal
# 3. Element-wise Greater
# 4. Element-wise Greater Equal
# 5. Element-wise Less
# 6. Element-wise Less Equal

# ~~~~~~~~~~~~~~~~~~~~~~~
# miscellaneous functions
# ~~~~~~~~~~~~~~~~~~~~~~~

# TODO:
# 1. Element-wise Negate
# 2. Element-wise Reciprocal
# 3. Element-wise Abs
# 4. Element-wise Sqrt
# 5. Element-wise Sq
# 6. Element-wise Ln
# 7. Element-wise Exp
# 8. Element-wise Sin
# 9. Element-wise Clip (?)

# ~~~~~~~~~~~~~~~~~~~~
# activation functions
# ~~~~~~~~~~~~~~~~~~~~

# TODO:
# 1. Element-wise Identity
# 2. Element-wise ReLu
# 3. Element-wise Sigmoid
# 4. Element-wise Tanh
# 5. Element-wise Leaky ReLu (?)
# 6. Element-wise Swish (?)
# 7. Element-wise GELU (?)
# 8. Element-wise ELU (?)