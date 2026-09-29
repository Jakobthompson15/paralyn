"""FP32 tensors and the paralyn.msl.tensor operator provider (versioned C ABI).

A tensor is an explicit (buffer, byte offset, shape, strides) descriptor; v1
operators require contiguous row-major FP32. matmul/bias_add/relu enqueue real
GPU work on the tensor's context queue and allocate distinct outputs. There is
no broadcasting beyond the documented bias row broadcast, no implicit context
and no CPU fallback. This is not cuBLAS, BLAS, NumPy or PyTorch compatibility.
"""
from array import array as _host_array
import ctypes as _c
from contextlib import ExitStack
from dataclasses import dataclass
from enum import IntEnum
import os

from . import (Access, Context, Error, Event, Module, Queue, Status, Type, _integer, _library,
               f32)

float32 = "float32"
TENSOR_VERSION_1 = 1
MAX_RANK = 8
_MAX_ELEMENTS = (1 << 31) - 1


class Activation(IntEnum):
    NONE = 0
    RELU = 1
    GELU_TANH = 2  # 0.5*x*(1 + tanh(0.7978845608028654f*(x + 0.044715f*x^3))); MSL has no erf


class Reduce(IntEnum):
    SUM = 0
    MAX = 1


class _TensorDesc(_c.Structure):
    _fields_ = [("struct_size", _c.c_uint32), ("version", _c.c_uint32),
                ("buffer", _c.c_uint64), ("byte_offset", _c.c_uint64),
                ("dtype", _c.c_int), ("rank", _c.c_uint32),
                ("shape", _c.c_uint64 * MAX_RANK), ("strides", _c.c_int64 * MAX_RANK)]


class _Matmul(_c.Structure):
    _fields_ = [("struct_size", _c.c_uint32), ("version", _c.c_uint32),
                ("m", _c.c_uint64), ("n", _c.c_uint64), ("k", _c.c_uint64),
                ("transpose_a", _c.c_uint32), ("transpose_b", _c.c_uint32),
                ("a", _c.POINTER(_TensorDesc)), ("b", _c.POINTER(_TensorDesc)),
                ("c", _c.POINTER(_TensorDesc))]


class _BiasActivation(_c.Structure):
    _fields_ = [("struct_size", _c.c_uint32), ("version", _c.c_uint32),
                ("activation", _c.c_int), ("reserved", _c.c_uint32),
                ("x", _c.POINTER(_TensorDesc)), ("bias", _c.POINTER(_TensorDesc)),
                ("out", _c.POINTER(_TensorDesc))]


class _BatchedMatmul(_c.Structure):
    _fields_ = [("struct_size", _c.c_uint32), ("version", _c.c_uint32),
                ("batch", _c.c_uint64), ("m", _c.c_uint64), ("n", _c.c_uint64), ("k", _c.c_uint64),
                ("a", _c.POINTER(_TensorDesc)), ("b", _c.POINTER(_TensorDesc)),
                ("c", _c.POINTER(_TensorDesc))]


class _ReduceRows(_c.Structure):
    _fields_ = [("struct_size", _c.c_uint32), ("version", _c.c_uint32),
                ("op", _c.c_int), ("reserved", _c.c_uint32),
                ("x", _c.POINTER(_TensorDesc)), ("out", _c.POINTER(_TensorDesc))]


class _SoftmaxRows(_c.Structure):
    _fields_ = [("struct_size", _c.c_uint32), ("version", _c.c_uint32),
                ("causal", _c.c_uint32), ("scale", _c.c_float),
                ("x", _c.POINTER(_TensorDesc)), ("out", _c.POINTER(_TensorDesc))]


class _LayerNorm(_c.Structure):
    _fields_ = [("struct_size", _c.c_uint32), ("version", _c.c_uint32),
                ("epsilon", _c.c_float), ("reserved", _c.c_uint32),
                ("x", _c.POINTER(_TensorDesc)), ("gamma", _c.POINTER(_TensorDesc)),
                ("beta", _c.POINTER(_TensorDesc)), ("out", _c.POINTER(_TensorDesc))]


class _Capabilities(_c.Structure):
    _fields_ = [("struct_size", _c.c_uint32), ("version", _c.c_uint32),
                ("operations", _c.c_uint64), ("activations", _c.c_uint64),
                ("reductions", _c.c_uint64), ("max_tensor_elements", _c.c_uint64),
                ("max_row_operator_rows", _c.c_uint64), ("max_layer_norm_columns", _c.c_uint64)]


class _Add(_c.Structure):
    _fields_ = [("struct_size", _c.c_uint32), ("version", _c.c_uint32),
                ("x", _c.POINTER(_TensorDesc)), ("y", _c.POINTER(_TensorDesc)),
                ("out", _c.POINTER(_TensorDesc))]


def _api(lib):
    if not getattr(lib, "_tensor_signatures", False):
        h, u32, u64, ptr = _c.c_uint64, _c.c_uint32, _c.c_uint64, _c.POINTER
        signatures = {
            "pr_tensor_desc_contiguous": (_c.c_int, [h, u64, _c.c_int, u32, ptr(u64), ptr(_TensorDesc)]),
            "pr_tensor_desc_validate": (_c.c_int, [ptr(_TensorDesc), ptr(u64)]),
            "pr_tensor_operators_artifact": (_c.c_int, [_c.c_void_p, u64, ptr(u64)]),
            "pr_tensor_operators_load": (_c.c_int, [h, ptr(h)]),
            "pr_matmul_f32": (_c.c_int, [h, h, ptr(_Matmul), ptr(h)]),
            "pr_bias_activation_f32": (_c.c_int, [h, h, ptr(_BiasActivation), ptr(h)]),
            "pr_batched_matmul_f32": (_c.c_int, [h, h, ptr(_BatchedMatmul), ptr(h)]),
            "pr_reduce_rows_f32": (_c.c_int, [h, h, ptr(_ReduceRows), ptr(h)]),
            "pr_softmax_rows_f32": (_c.c_int, [h, h, ptr(_SoftmaxRows), ptr(h)]),
            "pr_layer_norm_f32": (_c.c_int, [h, h, ptr(_LayerNorm), ptr(h)]),
            "pr_add_f32": (_c.c_int, [h, h, ptr(_Add), ptr(h)]),
            "pr_tensor_operators_capabilities": (_c.c_int, [ptr(_Capabilities)]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(lib.api, name)
            function.restype, function.argtypes = result, arguments
        lib._tensor_signatures = True
    return lib


def _shape(shape):
    if isinstance(shape, int) and not isinstance(shape, bool):
        shape = (shape,)
    if not isinstance(shape, (tuple, list)) or len(shape) > MAX_RANK:
        raise ValueError(f"shape must be a tuple of at most {MAX_RANK} dimensions")
    result = tuple(_integer(d, 0, (1 << 63) - 1, "dimension") for d in shape)
    count = 1
    for d in result:
        count *= d
    if count > _MAX_ELEMENTS:
        raise ValueError("FP32 tensors support at most INT32_MAX elements")
    return result, count


@dataclass(frozen=True)
class TensorDescriptor:
    """Low-level versioned descriptor over a Buffer. strides are in elements; None
    selects contiguous row-major. Operators reject other layouts explicitly."""
    buffer: object
    shape: tuple
    byte_offset: int = 0
    strides: object = None

    def _native(self):
        lib = _api(self.buffer._lib)
        shape = tuple(self.shape)
        if len(shape) > MAX_RANK:
            raise ValueError(f"rank exceeds {MAX_RANK}")
        desc = _TensorDesc()
        dims = (_c.c_uint64 * max(1, len(shape)))(*shape)
        lib.check(lib.api.pr_tensor_desc_contiguous(self.buffer.handle, self.byte_offset, Type.F32,
                                                    len(shape), dims, _c.byref(desc)))
        if self.strides is not None:
            if len(self.strides) != len(shape):
                raise ValueError("strides must have one entry per dimension")
            for index, stride in enumerate(self.strides):
                desc.strides[index] = _integer(stride, -(1 << 63), (1 << 63) - 1, "stride")
        return desc

    def required_bytes(self):
        lib = _api(self.buffer._lib)
        result = _c.c_uint64()
        lib.check(lib.api.pr_tensor_desc_validate(_c.byref(self._native()), _c.byref(result)))
        return result.value


def tensor_operators_artifact(library=None):
    """Exact PARALYNX1 bytes of the provider (for hashing/inspection/evidence)."""
    lib = _api(_library(library))
    size = _c.c_uint64()
    lib.check(lib.api.pr_tensor_operators_artifact(None, 0, _c.byref(size)))
    data = _c.create_string_buffer(size.value)
    lib.check(lib.api.pr_tensor_operators_artifact(data, size.value, _c.byref(size)))
    return data.raw[:size.value]


_OPERATION_BITS = ("matmul", "bias_activation", "batched_matmul", "reduce_rows", "softmax_rows",
                   "layer_norm", "add")


@dataclass(frozen=True)
class TensorOperatorCapabilities:
    """What this library's paralyn.msl.tensor provider implements and the shape
    limits it enforces (pr_tensor_operators_capabilities). A library property,
    not device availability: execution still needs a Metal device."""
    operations: frozenset
    activations: frozenset
    reductions: frozenset
    max_tensor_elements: int
    max_row_operator_rows: int
    max_layer_norm_columns: int


def _members(enum, mask):
    known = frozenset(v for v in enum if mask >> int(v) & 1)
    extra = mask & ~sum(1 << int(v) for v in known)
    if extra:  # a newer library than these bindings; never silently dropped
        raise Error(Status.UNSUPPORTED, "tensor_operators_capabilities",
                    f"unknown {enum.__name__} bits 0x{extra:x}; update the Python bindings")
    return known


def tensor_operators_capabilities(library=None):
    """Versioned capability record of the tensor operator provider; e.g.
    Activation.GELU_TANH in caps.activations detects GELU without trial calls."""
    lib = _api(_library(library))
    caps = _Capabilities(_c.sizeof(_Capabilities), TENSOR_VERSION_1)
    lib.check(lib.api.pr_tensor_operators_capabilities(_c.byref(caps)))
    operations = frozenset(name for bit, name in enumerate(_OPERATION_BITS) if caps.operations >> bit & 1)
    if caps.operations >> len(_OPERATION_BITS):
        raise Error(Status.UNSUPPORTED, "tensor_operators_capabilities",
                    f"unknown operation bits 0x{caps.operations:x}; update the Python bindings")
    return TensorOperatorCapabilities(operations, _members(Activation, caps.activations),
                                      _members(Reduce, caps.reductions), caps.max_tensor_elements,
                                      caps.max_row_operator_rows, caps.max_layer_norm_columns)


def load_tensor_operators(context):
    """Load the provider through the validated module path; caller owns the Module."""
    if not isinstance(context, Context):
        raise TypeError("load_tensor_operators requires a Context")
    lib = _api(context._lib)
    result = _c.c_uint64()
    lib.check(lib.api.pr_tensor_operators_load(context.handle, _c.byref(result)))
    return Module._adopt(lib, result.value)


def _event(lib, handle):
    return Event._adopt(lib, handle) if handle else None


def _flag(value, label):
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a bool")
    return int(value)


def matmul_into(queue, operators, a, b, c, *, m, n, k, transpose_a=False, transpose_b=False):
    """C ABI matmul with explicit m, n, k; returns an Event, or None when m*n == 0."""
    if not isinstance(queue, Queue) or not isinstance(operators, Module):
        raise TypeError("matmul_into requires a Queue and the tensor operator Module")
    lib = _api(queue._lib)
    descs = [x._native() for x in (a, b, c)]
    op = _Matmul(_c.sizeof(_Matmul), TENSOR_VERSION_1,
                 _integer(m, 0, (1 << 64) - 1, "m"), _integer(n, 0, (1 << 64) - 1, "n"),
                 _integer(k, 0, (1 << 64) - 1, "k"), _flag(transpose_a, "transpose_a"),
                 _flag(transpose_b, "transpose_b"), *(_c.pointer(d) for d in descs))
    result = _c.c_uint64()
    lib.check(lib.api.pr_matmul_f32(queue.handle, operators.handle, _c.byref(op), _c.byref(result)))
    return _event(lib, result.value)


def bias_activation_into(queue, operators, x, bias, out, *, activation=Activation.NONE):
    """out = act(x + bias[column]); bias may be None. Returns an Event or None if empty."""
    if not isinstance(queue, Queue) or not isinstance(operators, Module):
        raise TypeError("bias_activation_into requires a Queue and the tensor operator Module")
    lib = _api(queue._lib)
    xd, od = x._native(), out._native()
    bd = bias._native() if bias is not None else None
    op = _BiasActivation(_c.sizeof(_BiasActivation), TENSOR_VERSION_1, int(Activation(activation)), 0,
                         _c.pointer(xd), _c.pointer(bd) if bd is not None else None, _c.pointer(od))
    result = _c.c_uint64()
    lib.check(lib.api.pr_bias_activation_f32(queue.handle, operators.handle, _c.byref(op),
                                             _c.byref(result)))
    return _event(lib, result.value)


def _real(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    return float(value)


def _submit(queue, operators, name, record):
    if not isinstance(queue, Queue) or not isinstance(operators, Module):
        raise TypeError(f"{name} requires a Queue and the tensor operator Module")
    lib = _api(queue._lib)
    result = _c.c_uint64()
    lib.check(getattr(lib.api, name)(queue.handle, operators.handle, _c.byref(record), _c.byref(result)))
    return _event(lib, result.value)


def batched_matmul_into(queue, operators, a, b, c, *, batch, m, n, k):
    """C[i] = A[i] B[i] over rank-3 descriptors that may be strided (transposes and
    head splits as views). Returns an Event, or None when batch*m*n == 0."""
    descs = [x._native() for x in (a, b, c)]
    op = _BatchedMatmul(_c.sizeof(_BatchedMatmul), TENSOR_VERSION_1,
                        *(_integer(v, 0, (1 << 64) - 1, label)
                          for v, label in ((batch, "batch"), (m, "m"), (n, "n"), (k, "k"))),
                        *(_c.pointer(d) for d in descs))
    return _submit(queue, operators, "pr_batched_matmul_f32", op)


def reduce_rows_into(queue, operators, x, out, *, op=Reduce.SUM):
    """out[i] = sum or max over row i of x [rows, columns] (fixed documented order)."""
    xd, od = x._native(), out._native()
    record = _ReduceRows(_c.sizeof(_ReduceRows), TENSOR_VERSION_1, int(Reduce(op)), 0,
                         _c.pointer(xd), _c.pointer(od))
    return _submit(queue, operators, "pr_reduce_rows_f32", record)


def softmax_rows_into(queue, operators, x, out, *, scale=1.0, causal=False):
    """Row softmax of scale*x over the last dimension (rank 2 or 3); causal masks
    column j of row i when j > i + columns - rows."""
    xd, od = x._native(), out._native()
    record = _SoftmaxRows(_c.sizeof(_SoftmaxRows), TENSOR_VERSION_1, _flag(causal, "causal"),
                          _real(scale, "scale"), _c.pointer(xd), _c.pointer(od))
    return _submit(queue, operators, "pr_softmax_rows_f32", record)


def layer_norm_into(queue, operators, x, gamma, beta, out, *, epsilon=1e-5):
    """LayerNorm over the last dimension of x [rows, columns] with gamma/beta [columns]."""
    descs = [d._native() for d in (x, gamma, beta, out)]
    record = _LayerNorm(_c.sizeof(_LayerNorm), TENSOR_VERSION_1, _real(epsilon, "epsilon"), 0,
                        *(_c.pointer(d) for d in descs))
    return _submit(queue, operators, "pr_layer_norm_f32", record)


def add_into(queue, operators, x, y, out):
    """Elementwise out = x + y for equal-shape contiguous descriptors."""
    descs = [d._native() for d in (x, y, out)]
    record = _Add(_c.sizeof(_Add), TENSOR_VERSION_1, *(_c.pointer(d) for d in descs))
    return _submit(queue, operators, "pr_add_f32", record)


class _Operators:
    def __init__(self, context):
        with ExitStack() as acquired:
            self.module = acquired.enter_context(load_tensor_operators(context))
            self.queue = acquired.enter_context(context.queue())
            self._owners = acquired.pop_all()

    def close(self):
        self._owners.close()


def _operators(context):
    context.handle
    resources = getattr(context, "_tensor_resources", None)
    if resources is None:
        resources = _Operators(context)
        context._tensor_resources = resources
    return resources


class Tensor:
    """Owned contiguous row-major FP32 tensor on one explicit Context."""

    def __init__(self, *_, **__):
        raise TypeError("Use paralyn.tensor(...) or tensor operators to create a Tensor")

    @classmethod
    def _adopt(cls, context, buffer, shape, event=None, device=None):
        result = cls.__new__(cls)
        result._context, result._buffer, result._event = context, buffer, event
        result._shape, result._closed = shape, False
        result._device = context.device if device is None else device
        return result

    def _open(self):
        if self.closed:
            raise Error(Status.INVALID_HANDLE, "Tensor", "tensor is closed")

    @property
    def closed(self):
        return getattr(self, "_closed", True)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        count = 1
        for d in self._shape:
            count *= d
        return count

    @property
    def dtype(self):
        return float32

    @property
    def nbytes(self):
        return self.size * 4

    @property
    def device(self):
        return self._device

    def descriptor(self):
        self._open()
        return TensorDescriptor(self._buffer, self._shape)

    def wait(self):
        """Uploads and empty results have no producing GPU event and return None."""
        self._open()
        return self._event.wait() if self._event is not None else None

    def timing(self):
        self._open()
        return self._event.timing() if self._event is not None else None

    def to_host(self):
        """Flat row-major array('f') copy; see .shape for its layout."""
        self.wait()
        result = _host_array("f")
        result.frombytes(self._buffer.read())
        return result

    def close(self):
        if self.closed:
            return
        self._closed = True
        failure = None
        if self._event is not None:
            try:
                self._event.close()
            except Exception as error:
                failure = error
        try:
            self._buffer.close()
        except Exception as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self, _write=os.write, _exit=os._exit):
        if not self.closed:
            try:
                self.close()
            except Exception as error:
                try:
                    _write(2, f"Paralyn fatal unhandled Tensor finalizer error: {error}\n".encode())
                finally:
                    _exit(1)


def tensor(values, *, shape, context, dtype=float32):
    """Copy flat row-major values (sequence, or native-endian FP32 C-contiguous buffer)."""
    if not isinstance(context, Context):
        raise TypeError("tensor requires an explicit Context")
    context.handle
    if dtype != float32:
        raise ValueError("only explicit dtype=paralyn.float32 is supported")
    shape, count = _shape(shape)
    if isinstance(values, (list, tuple)):
        host = _host_array("f", (f32(value).value for value in values))
    else:
        view = memoryview(values)
        if not view.c_contiguous or view.ndim != 1:
            raise ValueError("tensor requires a flat C-contiguous host buffer")
        if view.itemsize != 4 or view.format not in ("f", "@f", "=f"):
            raise TypeError("buffer input must already contain native-endian FP32 values")
        host = _host_array("f")
        host.frombytes(view.tobytes())
    if len(host) != count:
        raise ValueError("value count must equal the product of the shape")
    data = host.tobytes()
    buffer = context.buffer(len(data))
    try:
        buffer.write(data)
        return Tensor._adopt(context, buffer, shape)
    except BaseException:
        buffer.close()
        raise


def _same_context(*tensors):
    for value in tensors:
        if not isinstance(value, Tensor):
            raise TypeError("tensor operators require Paralyn Tensors")
        value._open()
    context = tensors[0]._context
    if any(value._context is not context for value in tensors):
        raise Error(Status.CONTEXT_MISMATCH, "tensor operation", "inputs belong to different contexts")
    context.handle
    return context


def _output(context, shape, device, enqueue):
    buffer = context.buffer(4 * _shape(shape)[1])
    event = None
    try:
        event = enqueue(TensorDescriptor(buffer, shape))
        return Tensor._adopt(context, buffer, shape, event, device)
    except BaseException:
        try:
            if event is not None:
                event.close()
        finally:
            buffer.close()
        raise


def matmul(a, b, *, transpose_a=False, transpose_b=False):
    """C = op(A) op(B) on the GPU; FP32 accumulation in increasing k order."""
    context = _same_context(a, b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul requires rank-2 tensors")
    _flag(transpose_a, "transpose_a")
    _flag(transpose_b, "transpose_b")
    m, k = (a.shape[1], a.shape[0]) if transpose_a else a.shape
    kb, n = (b.shape[1], b.shape[0]) if transpose_b else b.shape
    if k != kb:
        raise ValueError(f"matmul inner dimensions differ ({k} != {kb})")
    resources = _operators(context)
    return _output(context, (m, n), a.device, lambda c: matmul_into(
        resources.queue, resources.module, a.descriptor(), b.descriptor(), c, m=m, n=n, k=k,
        transpose_a=transpose_a, transpose_b=transpose_b))


def _bias_activation(x, bias, activation):
    context = _same_context(x, *(() if bias is None else (bias,)))
    resources = _operators(context)
    return _output(context, x.shape, x.device, lambda out: bias_activation_into(
        resources.queue, resources.module, x.descriptor(),
        None if bias is None else bias.descriptor(), out, activation=activation))


def bias_add(x, bias, *, relu=False):
    """out[i, j] = x[i, j] + bias[j], optionally followed by ReLU, on the GPU."""
    if not isinstance(relu, bool):
        raise TypeError("relu must be a bool")
    return _bias_activation(x, bias, Activation.RELU if relu else Activation.NONE)


def relu(x):
    """out = x where x is not less than zero (NaN/-0.0 pass through), else +0.0."""
    return _bias_activation(x, None, Activation.RELU)


def gelu(x):
    """Tanh-form GELU on the GPU (see Activation.GELU_TANH)."""
    return _bias_activation(x, None, Activation.GELU_TANH)


def bias_gelu(x, bias):
    """out[i, j] = gelu_tanh(x[i, j] + bias[j]) on the GPU."""
    return _bias_activation(x, bias, Activation.GELU_TANH)


def batched_matmul(a, b):
    """C[i] = A[i] B[i] for contiguous rank-3 tensors [batch,m,k] x [batch,k,n]."""
    context = _same_context(a, b)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("batched_matmul requires rank-3 tensors")
    batch, m, k = a.shape
    if b.shape[0] != batch or b.shape[1] != k:
        raise ValueError(f"batched_matmul shapes differ: {a.shape} x {b.shape}")
    n = b.shape[2]
    resources = _operators(context)
    return _output(context, (batch, m, n), a.device, lambda c: batched_matmul_into(
        resources.queue, resources.module, a.descriptor(), b.descriptor(), c,
        batch=batch, m=m, n=n, k=k))


def _reduce(x, op):
    context = _same_context(x)
    if x.ndim != 2:
        raise ValueError("row reductions require rank-2 tensors")
    resources = _operators(context)
    return _output(context, (x.shape[0],), x.device, lambda out: reduce_rows_into(
        resources.queue, resources.module, x.descriptor(), out, op=op))


def row_sum(x):
    """Row sums of a rank-2 tensor in the documented fixed order."""
    return _reduce(x, Reduce.SUM)


def row_max(x):
    """Row maxima (NaN-propagating, +0.0 preferred over -0.0, -inf for empty rows)."""
    return _reduce(x, Reduce.MAX)


def softmax(x, *, scale=1.0, causal=False):
    """Numerically stable row softmax of scale*x over the last dimension."""
    context = _same_context(x)
    resources = _operators(context)
    return _output(context, x.shape, x.device, lambda out: softmax_rows_into(
        resources.queue, resources.module, x.descriptor(), out, scale=scale, causal=causal))


def layer_norm(x, gamma, beta, *, epsilon=1e-5):
    """LayerNorm over the last dimension of a rank-2 tensor."""
    context = _same_context(x, gamma, beta)
    resources = _operators(context)
    return _output(context, x.shape, x.device, lambda out: layer_norm_into(
        resources.queue, resources.module, x.descriptor(), gamma.descriptor(), beta.descriptor(),
        out, epsilon=epsilon))


def residual_add(x, y):
    """Elementwise x + y of equal-shape tensors (named apart from paralyn.add for Arrays)."""
    context = _same_context(x, y)
    resources = _operators(context)
    return _output(context, x.shape, x.device, lambda out: add_into(
        resources.queue, resources.module, x.descriptor(), y.descriptor(), out))
