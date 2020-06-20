from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    B = te.placeholder(shape, dtype=dtype, name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = te.placeholder(shape, dtype=dtype, name="A")
    B = te.placeholder(shape, dtype=dtype, name="B")
    C = te.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = te.placeholder(shape, dtype=dtype, name="A")
    C = te.compute(A.shape, lambda *i: A(*i) + const_k)

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = te.placeholder(shape, dtype=dtype, name="A")
    C = te.compute(A.shape, lambda *i: A(*i) * const_k)

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = te.placeholder(shape, dtype=dtype, name="A")
    b = tvm.tir.const(0, dtype=A.dtype)
    C = te.compute(A.shape, lambda *i: te.max(A(*i), b))

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    A = te.placeholder(shape, name="A")
    grad = te.placeholder(shape, name="grad")
    C = te.compute(A.shape, lambda *i: tvm.tir.Select(A(*i) < 0, 0.0, grad(*i)))

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, grad, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    A = te.placeholder(shapeA, name="A")
    B = te.placeholder(shapeB, name="B")
    if not transposeA and not transposeB:
        k = te.reduce_axis((0, shapeA[1]))
        C = te.compute((shapeA[0], shapeB[1]), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
    elif transposeA and not transposeB:
        k = te.reduce_axis((0, shapeA[0]))
        C = te.compute((shapeA[1], shapeB[1]), lambda i, j: te.sum(A[k, i] * B[k, j], axis=k))
    elif not transposeA and transposeB:
        k = te.reduce_axis((0, shapeA[1]))
        C = te.compute((shapeA[0], shapeB[0]), lambda i, j: te.sum(A[i, k] * B[j, k], axis=k))
    elif transposeA and transposeB:
        k = te.reduce_axis((0, shapeA[0]))
        C = te.compute((shapeA[1], shapeB[0]), lambda i, j: te.sum(A[k, i] * B[j, k], axis=k))

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    A = te.placeholder(shapeX, name="A")
    B = te.placeholder(shapeF, name="B")
    out_shape = (N, M, H-R+1, W-S+1)

    dc = te.reduce_axis((0, C), name="dc")
    dh = te.reduce_axis((0, R), name="dh")
    dw = te.reduce_axis((0, S), name="dw")
    C = te.compute(out_shape, lambda n, m, i, j: te.sum(
        A[n, dc, i+dh, j+dw] * B[m, dc, dh, dw],
        axis=[dc, dh, dw])
    )

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    A = te.placeholder(shape, name="A")

    k = te.reduce_axis((0, shape[1]))
    max = te.compute((shape[0], ), lambda i: te.max(A[i, k], axis=k))
    e_x = te.compute(shape, lambda i, j: te.exp(A[i, j] - max[i]))

    q = te.reduce_axis((0, shape[1]))
    e_x_sum = te.compute((shape[0], ), lambda i: te.sum(e_x[i, q], axis=q))

    softmax = te.compute(shape, lambda i, j: e_x[i, j] / e_x_sum[i])

    s = te.create_schedule(softmax.op)
    f = tvm.build(s, [A, softmax], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""
    A = te.placeholder(shape, name="A")
    B = te.placeholder(shape, name="B")

    k = te.reduce_axis((0, shape[1]))
    max = te.compute((shape[0],), lambda i: te.max(A[i, k], axis=k))
    e_x = te.compute(shape, lambda i, j: te.exp(A[i, j] - max[i]))

    q = te.reduce_axis((0, shape[1]))
    e_x_sum = te.compute((shape[0],), lambda i: te.sum(e_x[i, q], axis=q))

    softmax = te.compute(shape, lambda i, j: e_x[i, j] / e_x_sum[i])

    p = te.reduce_axis((0, shape[1]))
    cross_entropy = te.compute((shape[0],), lambda i: te.sum(-1 * B[i, p] * te.log(softmax[i, p]), axis=p))

    t = te.reduce_axis((0, shape[0]))
    sce = te.compute((1, ), lambda i: te.sum(cross_entropy[t], axis=t))
    ace = te.compute((1, ), lambda i: sce[i] / shape[0])

    s = te.create_schedule(ace.op)
    f = tvm.build(s, [A, B, ace], tgt, target_host=tgt_host, name=func_name)
    return f


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = te.placeholder(shape, dtype=dtype, name="A")
    grad = te.placeholder(shape, dtype=dtype, name="grad")
    Y = te.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = te.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f