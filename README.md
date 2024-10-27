<!--
 Copyright (c) 2024 Garrett Kinman
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

A lightweight, dependency-free tensor library that uses zero dynamic memory allocation. Intended for microcontrollers.

# Concept
This library is part of the broader `tinyNN` framework for TinyML, which is built around a few key principles:
1. **No dynamic memory allocation.** This is to improve performance and to provide compile-time guarantees of memory usage. Currently no other TinyML frameworks (that I know of) use only static memory allocation.
2. **A small set of primitive operators.** All higher-level operations (e.g., matrix multiplication) can be broken down to some combination of these primitive operations. So long as these primitive operators (primops) are optimized for hardware, you will have reasonably performant higher-level operations.
3. **Portable to new hardware.** Porting to new hardware (including dedicated accelerators) is as easy as implementing the primops on the hardware. If that isn't efficient enough, you can implement a new tensor subtype to optimize to your heart's content.
4. **No dependencies.** Working with other TinyML frameworks can be a pain, as there are so many dependencies that can (and often do) give you problems. By avoiding dependencies, this framework is much easier to use, simpler to understand and debug, and less of a pain to set up and use.

# Hardware Acceleration
If you want to accelerate this library on custom hardware (e.g., vector/SIMD instructions), just create a new tensor subclass that inherits from `Tensor`, and override the primitive operators with your own optimized implementations.

**TODO:** add a code example

# Primitive Operators
1. Arithmetic Operations
   1. Element-wise Vector Addition
   2. Element-wise Vector Subtraction
   3. Element-wise Vector Multiplication
   4. Element-wise Vector Division
   5. Element-wise Vector Modulus (?)
2. Activation Functions
   1. Element-wise Identity
   2. Element-wise ReLu
   3. Element-wise Sigmoid
   4. Element-wise Tanh
   5. Element-wise Leaky ReLu (?)
   6. Element-wise Swish (?)
   7. Element-wise Softmax (???)
   8. More?
3. Reduction Operations
   1. Sum Reduction
   2. Mean Reduction
   3. Max Reduction
   4. Min Reduction
   5. Argmax Reduction
   6. Argmin Reduction
   7. More?
4. Logical Operations
   1. Element-wise Equal
   2. Element-wise Not Equal
   3. Element-wise Greater
   4. Element-wise Greater Equal
   5. Element-wise Less
   6. Element-wise Less Equal
5. Miscellaneous Functions
   1. Element-wise Abs
   2. Element-wise Sqrt
   3. Element-wise Pow
   4. Element-wise Ln
   5. Element-wise Exp
   6. Element-wise Sin
   7. Element-wise Cos
   8. Element-wise Negate
   9. Element-wise Floor (?)
   10. Element-wise Ceil (?)
   11. Element-wise Round (?)
   12. Element-wise Clip (?)
   13. More?
6. More?