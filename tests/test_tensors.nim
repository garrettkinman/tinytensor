# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import tinytensor

# TODO: add unit tests

test "basic test":
    # Create a 2x3 tensor of integers
    const tensorShape: TensorShape[2] = [2, 3]
    var t = initTensor[int, tensorShape]()

    # Set some values
    t[0, 0] = 1
    t[0, 1] = 2
    t[0, 2] = 3
    t[1, 0] = 4
    t[1, 1] = 5
    t[1, 2] = 6

    # Create another tensor and perform operations
    const doubleValue = proc(x: int): float {.noSideEffect.} = x.float * 2
    var t2 = t.map(doubleValue)
    echo t2  # Will print the tensor with doubled values

    # Perform element-wise addition
    const addValues = proc(x, y: int): int {.noSideEffect.} = x + y
    var t3 = zip(t, t, addValues)
    echo t3  # Will print the tensor with summed values